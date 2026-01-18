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
def benchmark_transformer(model_info, dtype) -> Iterator[benchmark.Timer]:
    device = 'cuda'
    model_name, model_factory, input_shape = model_info
    inp = torch.randn(input_shape, dtype=dtype, device=device)
    for mod_name, mod_apply in MODIFIERS:
        model: nn.Module = model_factory()
        model = mod_apply(model).to(device).to(dtype)
        out = model(inp)
        grad = out.clone()
        out.backward(grad)
        yield benchmark.Timer(stmt='model(inp).backward(grad)', globals={'model': model, 'inp': inp, 'grad': grad}, label='fw+bw', description=mod_name, sub_label=model_name)