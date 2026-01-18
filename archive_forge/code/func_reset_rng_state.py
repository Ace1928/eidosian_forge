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
def reset_rng_state(use_xla=False):
    torch.manual_seed(1337)
    random.seed(1337)
    if np:
        np.random.seed(1337)
    if use_xla:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(1337, str(xm.xla_device()))