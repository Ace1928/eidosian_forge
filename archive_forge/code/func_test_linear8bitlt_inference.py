import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
@pytest.mark.parametrize('threshold', [0.0, 3.0], ids=id_formatter('threshold'))
def test_linear8bitlt_inference(threshold):
    l1 = bnb.nn.Linear8bitLt(32, 64, threshold=threshold).cuda().half()
    assert l1.weight.device.type == 'cuda'
    assert l1.weight.dtype == torch.float16
    l1.eval()
    for i in range(100):
        b1 = torch.randn(16, 8, 32, device='cuda').half()
        o1 = l1(b1)
        if i == 1:
            assert l1.state.CxB is not None