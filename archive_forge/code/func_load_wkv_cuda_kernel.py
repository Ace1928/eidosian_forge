import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_rwkv import RwkvConfig
def load_wkv_cuda_kernel(context_length):
    from torch.utils.cpp_extension import load as load_kernel
    global rwkv_cuda_kernel
    kernel_folder = Path(__file__).resolve().parent.parent.parent / 'kernels' / 'rwkv'
    cuda_kernel_files = [kernel_folder / f for f in ['wkv_op.cpp', 'wkv_cuda.cu', 'wkv_cuda_bf16.cu']]
    if rwkv_cuda_kernel is not None and rwkv_cuda_kernel.max_seq_length == context_length:
        return
    logger.info(f'Loading CUDA kernel for RWKV at context length of {context_length}.')
    flags = ['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization', f'-DTmax={context_length}']
    rwkv_cuda_kernel = load_kernel(name=f'wkv_{context_length}', sources=cuda_kernel_files, verbose=logging.get_verbosity() == logging.DEBUG, extra_cuda_cflags=flags)
    rwkv_cuda_kernel.max_seq_length = context_length