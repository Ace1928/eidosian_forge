import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator
import torch
from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.consts import NONPYTORCH_DOC_URL
from bitsandbytes.cuda_specs import CUDASpecs
from bitsandbytes.diagnostics.utils import print_dedented
def print_cuda_diagnostics(cuda_specs: CUDASpecs) -> None:
    print(f'PyTorch settings found: CUDA_VERSION={cuda_specs.cuda_version_string}, Highest Compute Capability: {cuda_specs.highest_compute_capability}.')
    binary_path = get_cuda_bnb_library_path(cuda_specs)
    if not binary_path.exists():
        print_dedented(f'\n        Library not found: {binary_path}. Maybe you need to compile it from source?\n        If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION`,\n        for example, `make CUDA_VERSION=113`.\n\n        The CUDA version for the compile might depend on your conda install, if using conda.\n        Inspect CUDA version via `conda list | grep cuda`.\n        ')
    cuda_major, cuda_minor = cuda_specs.cuda_version_tuple
    if cuda_major < 11:
        print_dedented('\n            WARNING: CUDA versions lower than 11 are currently not supported for LLM.int8().\n            You will be only to use 8-bit optimizers and quantization routines!\n            ')
    print(f'To manually override the PyTorch CUDA version please see: {NONPYTORCH_DOC_URL}')
    if not cuda_specs.has_cublaslt:
        print_dedented('\n            WARNING: Compute capability < 7.5 detected! Only slow 8-bit matmul is supported for your GPU!\n            If you run into issues with 8-bit matmul, you can try 4-bit quantization:\n            https://huggingface.co/blog/4bit-transformers-bitsandbytes\n            ')