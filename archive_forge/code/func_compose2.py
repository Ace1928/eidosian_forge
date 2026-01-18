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
def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))