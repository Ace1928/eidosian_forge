import logging
from typing import Union
import torch
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
@staticmethod
def create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
    try:
        from scipy.stats import norm
    except ImportError:
        raise ImportError("The required package 'scipy' is not installed. Please install it to continue.")
    variations = 2 ** num_bits
    if symmetric:
        v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
        values = []
        for index in range(len(v) - 1):
            values.append(0.5 * v[index] + 0.5 * v[index + 1])
        v = values
    else:
        v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
        v2 = [0]
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
        v = v1 + v2 + v3
    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()
    return values