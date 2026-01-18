import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def check_matmul(A, B, out, transposed_A, transposed_B, expected_type=torch.int8):
    if not torch.cuda.is_initialized():
        torch.cuda.init()
    if A.dtype != expected_type or B.dtype != expected_type:
        raise TypeError(f'Expected torch.int8 input tensors A and B, but got {A.dtype} and {B.dtype}')
    sA = A.shape
    sB = B.shape
    tA = transposed_A
    tB = transposed_B
    correct = True
    if len(sA) == 2 and len(sB) == 2:
        if not tA and (not tB) and (A.shape[1] != B.shape[0]):
            correct = False
        elif tA and (not tB) and (A.shape[0] != B.shape[0]):
            correct = False
        elif tA and tB and (A.shape[0] != B.shape[1]):
            correct = False
        elif not tA and tB and (A.shape[1] != B.shape[1]):
            correct = False
    elif len(sA) == 3 and len(sB) == 2:
        if not tA and (not tB) and (A.shape[2] != B.shape[0]):
            correct = False
        elif tA and (not tB) and (A.shape[1] != B.shape[0]):
            correct = False
        elif tA and tB and (A.shape[1] != B.shape[1]):
            correct = False
        elif not tA and tB and (A.shape[2] != B.shape[1]):
            correct = False
    elif len(sA) == 3 and len(sB) == 3:
        if not tA and (not tB) and (A.shape[2] != B.shape[1]):
            correct = False
        elif tA and (not tB) and (A.shape[1] != B.shape[1]):
            correct = False
        elif tA and tB and (A.shape[1] != B.shape[2]):
            correct = False
        elif not tA and tB and (A.shape[2] != B.shape[2]):
            correct = False
    if out is not None:
        sout = out.shape
        if not correct and len(sA) == 3 and (len(sB) == 3):
            if sout[0] == sA[2] and sout[1] == sB[2] and (sA[0] == sB[0]) and (sA[1] == sB[1]):
                correct = True
    elif len(sA) == 2 and len(sB) == 2:
        if not tA and (not tB):
            sout = (sA[0], sB[1])
        elif tA and tB:
            sout = (sA[1], sB[0])
        elif tA and (not tB):
            sout = (sA[1], sB[1])
        elif not tA and tB:
            sout = (sA[0], sB[0])
    elif len(sA) == 3 and len(sB) == 2:
        if not tA and (not tB):
            sout = (sA[0], sA[1], sB[1])
        elif tA and tB:
            sout = (sA[0], sA[2], sB[0])
        elif tA and (not tB):
            sout = (sA[0], sA[2], sB[1])
        elif not tA and tB:
            sout = (sA[0], sA[1], sB[0])
    elif len(sA) == 3 and len(sB) == 3:
        if not tA and (not tB):
            sout = (sA[0], sA[1], sB[2])
        elif tA and tB:
            sout = (sA[0], sA[2], sB[1])
        elif tA and (not tB):
            sout = (sA[0], sA[2], sB[2])
        elif not tA and tB:
            sout = (sA[0], sA[1], sB[1])
    if not correct:
        raise ValueError(f'Tensor dimensions incorrect for matrix mulitiplication: A x B: {sA} x {sB} with transpose for A x B: {tA} x {tB}.')
    return sout