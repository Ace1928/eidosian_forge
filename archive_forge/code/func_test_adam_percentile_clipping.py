import os
from os.path import join
import shutil
import time
import uuid
from lion_pytorch import Lion
import pytest
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F
from tests.helpers import describe_dtype, id_formatter
@pytest.mark.parametrize('optim_bits', [32, 8], ids=id_formatter('optim_bits'))
@pytest.mark.parametrize('gtype', [torch.float32], ids=describe_dtype)
@pytest.mark.parametrize('dim2', [32, 1024, 4097], ids=id_formatter('dim2'))
@pytest.mark.parametrize('dim1', [1024], ids=id_formatter('dim1'))
def test_adam_percentile_clipping(dim1, dim2, gtype, optim_bits):
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device='cpu', dtype=gtype) * 0.1
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-08
    p1 = p1.cuda()
    p2 = p1.clone()
    adam1 = bnb.optim.Adam([p1], lr, (beta1, beta2), eps, optim_bits=optim_bits)
    adam2 = bnb.optim.Adam([p2], lr, (beta1, beta2), eps, optim_bits=optim_bits, percentile_clipping=5)
    gnorm_vec = torch.zeros(100).cuda()
    step = 0
    for i in range(50):
        step += 1
        g1 = torch.randn(dim1, dim2, device='cuda', dtype=gtype) * 0.1 + 0.01 * i
        g2 = g1.clone()
        p2.grad = g2
        current_gnorm, clip_val, gnorm_scale = F.percentile_clipping(g1, gnorm_vec, step, 5)
        g1 = (g1.float() * gnorm_scale).to(gtype)
        p1.grad = g1
        adam1.step()
        adam2.step()
        if optim_bits == 32:
            torch.testing.assert_close(p1, p2)
            torch.testing.assert_close(adam1.state[p1]['state1'], adam2.state[p2]['state1'], atol=5e-05, rtol=0.0001)
            torch.testing.assert_close(adam1.state[p1]['state2'], adam2.state[p2]['state2'], atol=5e-05, rtol=0.0001)
        elif optim_bits == 8:
            torch.testing.assert_close(p1, p2, atol=0.0001, rtol=0.001)
            torch.testing.assert_close(adam1.state[p1]['state1'], adam2.state[p2]['state1'], atol=2, rtol=0.001)
            torch.testing.assert_close(adam1.state[p1]['state2'], adam2.state[p2]['state2'], atol=2, rtol=0.001)
            adam1.state[p1]['state1'].copy_(adam2.state[p2]['state1'])
            adam1.state[p1]['state2'].copy_(adam2.state[p2]['state2'])
        if i % 10 == 0 and i > 0:
            path = get_temp_dir()
            torch.save(adam2.state_dict(), join(path, 'opt.pt'))
            del adam2
            adam2 = None
            adam2 = bnb.optim.Adam([p2], lr, (beta1, beta2), eps, optim_bits=optim_bits, percentile_clipping=5)
            adam2.load_state_dict(torch.load(join(path, 'opt.pt')))