import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator
import torch
from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.consts import NONPYTORCH_DOC_URL
from bitsandbytes.cuda_specs import CUDASpecs
from bitsandbytes.diagnostics.utils import print_dedented
def print_cuda_runtime_diagnostics() -> None:
    cudart_paths = list(find_cudart_libraries())
    if not cudart_paths:
        print('CUDA SETUP: WARNING! CUDA runtime files not found in any environmental path.')
    elif len(cudart_paths) > 1:
        print_dedented(f'\n            Found duplicate CUDA runtime files (see below).\n\n            We select the PyTorch default CUDA runtime, which is {torch.version.cuda},\n            but this might mismatch with the CUDA version that is needed for bitsandbytes.\n            To override this behavior set the `BNB_CUDA_VERSION=<version string, e.g. 122>` environmental variable.\n\n            For example, if you want to use the CUDA version 122,\n                BNB_CUDA_VERSION=122 python ...\n\n            OR set the environmental variable in your .bashrc:\n                export BNB_CUDA_VERSION=122\n\n            In the case of a manual override, make sure you set LD_LIBRARY_PATH, e.g.\n            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2,\n            ')
        for pth in cudart_paths:
            print(f'* Found CUDA runtime at: {pth}')