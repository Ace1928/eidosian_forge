import argparse
import json
import os
import re
import torch
from transformers import BloomConfig, BloomModel
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging
def get_dtype_size(dtype):
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search('[^\\d](\\d+)$', str(dtype))
    if bit_search is None:
        raise ValueError(f'`dtype` is not a valid dtype: {dtype}.')
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8