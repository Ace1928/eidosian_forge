from contextlib import nullcontext
import os
from tempfile import TemporaryDirectory
import pytest
import torch
import bitsandbytes as bnb
from bitsandbytes import functional as F
from bitsandbytes.autograd import get_inverse_transform_indices, undo_layout
from bitsandbytes.nn.modules import Linear8bitLt
from tests.helpers import (
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5), reason='this test requires a turing-generation or newer GPU, see bitsandbytes docs')
def test_layout_exact_match():
    x = (torch.randn(14336 * 3, 14336) * 10).to(torch.int8).cuda()
    for tile_size, order in (((8, 32), 'col_turing'), ((32, 32), 'col_ampere')):
        transform = lambda x: F.transform(x.cuda(), from_order='row', to_order=order)[0].to(x.device)
        tile_indices = get_inverse_transform_indices(transform, tile_size)
        cxb = transform(x)
        torch.cuda.synchronize()
        restored_x = undo_layout(cxb, tile_indices)
        torch.cuda.synchronize()
        assert restored_x.is_contiguous()
        assert torch.all(torch.eq(restored_x, x))