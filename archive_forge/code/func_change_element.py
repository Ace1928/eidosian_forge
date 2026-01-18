import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def change_element(tup, index, value):
    ls = list(tup)
    ls[index] = value
    return tuple(ls)