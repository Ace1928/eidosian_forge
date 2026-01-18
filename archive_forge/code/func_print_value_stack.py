import builtins
import dis
import traceback
from typing import Optional, Union
import torch
from .exc import unimplemented
@staticmethod
def print_value_stack(*, stacklevel=0):
    comptime(lambda ctx: ctx.print_value_stack(stacklevel=ctx.get_local('stacklevel').as_python_constant() + 1))