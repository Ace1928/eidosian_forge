import torch
import random
import os
import queue
from dataclasses import dataclass
from torch._utils import ExceptionWrapper
from typing import Optional, Union, TYPE_CHECKING
from . import signal_handling, MP_STATUS_CHECK_INTERVAL, IS_WINDOWS, HAS_NUMPY
def mix(x, y):
    result_x = MIX_MULT_L * x & MASK32
    result_y = MIX_MULT_R * y & MASK32
    result = result_x - result_y & MASK32
    result = (result ^ result >> XSHIFT) & MASK32
    return result