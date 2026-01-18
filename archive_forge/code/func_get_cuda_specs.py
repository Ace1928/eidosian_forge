import dataclasses
from typing import List, Optional, Tuple
import torch
def get_cuda_specs() -> Optional[CUDASpecs]:
    if not torch.cuda.is_available():
        return None
    return CUDASpecs(highest_compute_capability=get_compute_capabilities()[-1], cuda_version_string=get_cuda_version_string(), cuda_version_tuple=get_cuda_version_tuple())