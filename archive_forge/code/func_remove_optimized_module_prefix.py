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
def remove_optimized_module_prefix(name) -> str:
    return re.sub('^_orig_mod[.]', '', name)