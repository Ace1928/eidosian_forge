import itertools
from functools import partial, reduce
from typing import Iterator
import timm
import torch
import torch.nn as nn
from timm.models.layers import Mlp as TimmMlp
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock
from torch.utils import benchmark
import xformers.ops as xops
from xformers.benchmarks.utils import benchmark_main_helper
def replace_module(module: nn.Module, replace_class, factory):
    if isinstance(module, replace_class):
        return factory(module)
    module_output = module
    for name, child in module.named_children():
        module_output.add_module(name, replace_module(child, replace_class, factory))
    del module
    return module_output