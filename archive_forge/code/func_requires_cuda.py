import pytest
import torch
@pytest.fixture(scope='session')
def requires_cuda() -> bool:
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        pytest.skip('CUDA is required')
    return cuda_available