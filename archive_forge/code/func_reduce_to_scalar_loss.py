import contextlib
import dis
import functools
import logging
import os.path
import random
import re
import sys
import types
import unittest
from typing import List, Optional, Sequence, Union
from unittest.mock import patch
import torch
from torch import fx
from torch._dynamo.output_graph import OutputGraph
from . import config, eval_frame, optimize_assert, reset
from .bytecode_transformation import (
from .guards import CheckFunctionManager, GuardedCode
from .utils import same
def reduce_to_scalar_loss(out):
    """Reduce the output of a model to get scalar loss"""
    if isinstance(out, torch.Tensor):
        return out.sum() / out.numel()
    elif isinstance(out, (list, tuple)):
        return sum([reduce_to_scalar_loss(x) for x in out]) / len(out)
    elif type(out).__name__ in ('MaskedLMOutput', 'Seq2SeqLMOutput', 'CausalLMOutputWithCrossAttentions'):
        return reduce_to_scalar_loss(out.logits)
    elif type(out).__name__ == 'SquashedNormal':
        return out.mean.sum()
    elif isinstance(out, dict):
        return sum([reduce_to_scalar_loss(value) for value in out.values()]) / len(out.keys())
    raise NotImplementedError("Don't know how to reduce", type(out))