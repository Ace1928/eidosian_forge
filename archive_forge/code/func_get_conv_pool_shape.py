import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def get_conv_pool_shape(image_shape, args, out_ch, transpose):
    batch, in_c, in_h, in_w = image_shape
    if args.dilation_h != 1 or args.dilation_w != 1:
        raise Exception('Dilation not supported yet.')
    if transpose:
        out_h = (in_h - 1) * args.stride_h + args.kernel_h - args.pad_t - args.pad_b
        out_w = (in_w - 1) * args.stride_w + args.kernel_w - args.pad_l - args.pad_l
    else:
        out_h = (in_h - args.kernel_h + args.pad_t + args.pad_b) // args.stride_h + 1
        out_w = (in_w - args.kernel_w + args.pad_l + args.pad_r) // args.stride_w + 1
    if in_h == 0:
        out_h = 0
    if in_w == 0:
        out_w = 0
    out_shape = (batch, out_ch, out_h, out_w)
    return out_shape