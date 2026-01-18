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
def debug_dump(name, code: types.CodeType, extra='') -> None:
    with open(os.path.join(debug_dir(), name), 'w') as fd:
        fd.write(f'{dis.Bytecode(code).info()}\n\n{dis.Bytecode(code).dis()}\n\n{extra}\n')