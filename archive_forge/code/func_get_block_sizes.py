from functools import reduce  # Required in Python 3
import operator
from typing import Optional
import warnings
import torch
from bitsandbytes.autograd._functions import GlobalOutlierPooler, MatmulLtState
import bitsandbytes.functional as F
def get_block_sizes(input_matrix, weight_matrix):
    input_features = input_matrix.shape[-1]
    output_features = weight_matrix.shape[0] if weight_matrix.shape[1] == input_features else weight_matrix.shape[1]
    array = [4096, 2048, 1024, 512, 256, 128, 64, 0]
    bsz, bsz2 = (1024, 1024)
    for i, k in enumerate(array):
        if input_features > array[i + 1]:
            bsz = k
            break
    for i, k in enumerate(array):
        if output_features > array[i + 1]:
            bsz2 = k
            break
    return (bsz, bsz2)