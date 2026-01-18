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
def debug_insert_nops(frame, cache_size, hooks, _) -> Optional[GuardedCode]:
    """used to debug jump updates"""

    def insert_nops(instructions, code_options):
        instructions.insert(0, create_instruction('NOP'))
        instructions.insert(0, create_instruction('NOP'))
    if is_generator(frame.f_code):
        return None
    debug_checks(frame.f_code)
    code = transform_code_object(frame.f_code, insert_nops)
    graph = OutputGraph(code_options={}, compiler_fn=None, root_tx=None, export=False, export_constraints=None, frame_state={'_id': 0}, local_scope=locals(), global_scope=globals(), f_code=frame.f_code)
    return GuardedCode(code, CheckFunctionManager(graph).check_fn)