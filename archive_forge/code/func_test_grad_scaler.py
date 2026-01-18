import pytest
from hypothesis import given, settings
from hypothesis.strategies import lists, one_of, tuples
from thinc.api import PyTorchGradScaler
from thinc.compat import has_torch, has_torch_amp, has_torch_cuda_gpu, torch
from thinc.util import is_torch_array
from ..strategies import ndarrays
@pytest.mark.skipif(not has_torch, reason='needs PyTorch')
@pytest.mark.skipif(not has_torch_cuda_gpu, reason='needs a GPU')
@pytest.mark.skipif(not has_torch_amp, reason='requires PyTorch with mixed-precision support')
def test_grad_scaler():
    import torch
    device_id = torch.cuda.current_device()
    scaler = PyTorchGradScaler(enabled=True)
    scaler.to_(device_id)
    t = torch.tensor([1.0], device=device_id)
    assert scaler.scale([torch.tensor([1.0], device=device_id)]) == [torch.tensor([2.0 ** 16], device=device_id)]
    assert scaler.scale(torch.tensor([1.0], device=device_id)) == torch.tensor([2.0 ** 16], device=device_id)
    with pytest.raises(ValueError):
        scaler.scale('bogus')
    with pytest.raises(ValueError):
        scaler.scale(42)
    g = [torch.tensor([2.0 ** 16], device=device_id), torch.tensor([float('Inf')], device=device_id)]
    assert scaler.unscale(g)
    assert g[0] == torch.tensor([1.0]).cuda()
    scaler.update()
    assert scaler.scale([torch.tensor([1.0], device=device_id)]) == [torch.tensor([2.0 ** 15], device=device_id)]