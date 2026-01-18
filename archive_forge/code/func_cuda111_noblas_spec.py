import pytest
from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.cuda_specs import CUDASpecs
@pytest.fixture
def cuda111_noblas_spec() -> CUDASpecs:
    return CUDASpecs(cuda_version_string='111', highest_compute_capability=(7, 2), cuda_version_tuple=(11, 1))