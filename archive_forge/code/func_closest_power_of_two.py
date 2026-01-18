import os
import math
import sys
from typing import Optional, Union, Callable
import pyglet
from pyglet.customtypes import Buffer
def closest_power_of_two(x: int) -> int:
    if x <= 2:
        return 2
    if x >> x.bit_length() - 2 & 1:
        return 1 << math.ceil(math.log2(x))
    else:
        return 1 << math.floor(math.log2(x))