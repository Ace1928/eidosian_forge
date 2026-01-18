import pytest
from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.cuda_specs import CUDASpecs
def test_get_cuda_bnb_library_path_nocublaslt(monkeypatch, cuda111_noblas_spec):
    monkeypatch.delenv('BNB_CUDA_VERSION', raising=False)
    assert get_cuda_bnb_library_path(cuda111_noblas_spec).stem == 'libbitsandbytes_cuda111_nocublaslt'