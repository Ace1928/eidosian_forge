import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.quantization
import torch.ao.ns._numeric_suite as ns
def parent_child_names(name):
    """Split full name of submodule into parent submodule's full name and submodule's name."""
    split_name = name.rsplit('.', 1)
    if len(split_name) == 1:
        return ('', split_name[0])
    else:
        return (split_name[0], split_name[1])