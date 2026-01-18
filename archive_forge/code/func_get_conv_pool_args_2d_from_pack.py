import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def get_conv_pool_args_2d_from_pack(self, kernel_size, packed_config):
    pc = [i.item() for i in packed_config]
    assert pc[0] == 2
    strides = [pc[1], pc[2]]
    paddings = [pc[3], pc[4]]
    dilations = [pc[5], pc[6]]
    output_padding = [pc[7], pc[8]]
    group_num = pc[9]
    assert len(pc) == 11
    assert output_padding == [0, 0]
    return self.get_conv_pool_args_2d_common(kernel_size, strides, paddings, dilations, group_num)