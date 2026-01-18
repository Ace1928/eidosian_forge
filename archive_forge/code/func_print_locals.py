import builtins
import dis
import traceback
from typing import Optional, Union
import torch
from .exc import unimplemented
@staticmethod
def print_locals(*, stacklevel=0):
    comptime(lambda ctx: ctx.print_locals(stacklevel=ctx.get_local('stacklevel').as_python_constant() + 1))