import pytest
from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.cuda_specs import CUDASpecs
def test_get_cuda_bnb_library_path_override(monkeypatch, cuda120_spec, caplog):
    monkeypatch.setenv('BNB_CUDA_VERSION', '110')
    assert get_cuda_bnb_library_path(cuda120_spec).stem == 'libbitsandbytes_cuda110'
    assert 'BNB_CUDA_VERSION' in caplog.text