import builtins
import dis
import traceback
from typing import Optional, Union
import torch
from .exc import unimplemented
@staticmethod
def print_guards():
    comptime(lambda ctx: ctx.print_guards())