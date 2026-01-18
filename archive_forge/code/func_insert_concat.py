import collections
import copy
import functools
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
import torch
from transformers.file_utils import add_end_docstrings
from transformers.utils.fx import _gen_constructor_wrapper
def insert_concat(linear_input):
    shape = linear_input.shape[:-1] + (1,)
    return torch.cat([linear_input, torch_ones(shape, device=linear_input.device)], dim=-1)