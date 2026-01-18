import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent
from .... import base
def make_aux(classes):
    out = nn.HybridSequential(prefix='')
    out.add(nn.AvgPool2D(pool_size=5, strides=3))
    out.add(_make_basic_conv(channels=128, kernel_size=1))
    out.add(_make_basic_conv(channels=768, kernel_size=5))
    out.add(nn.Flatten())
    out.add(nn.Dense(classes))
    return out