import argparse
import os
import re
import zipfile
import torch
from transformers import AutoTokenizer, GPT2Config
def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    input_shape = param.size()
    if checkpoint_version == 1.0:
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param