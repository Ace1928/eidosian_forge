import sys
import types
import collections
import io
from opcode import *
from opcode import (
@classmethod
def from_traceback(cls, tb, *, show_caches=False, adaptive=False):
    """ Construct a Bytecode from the given traceback """
    while tb.tb_next:
        tb = tb.tb_next
    return cls(tb.tb_frame.f_code, current_offset=tb.tb_lasti, show_caches=show_caches, adaptive=adaptive)