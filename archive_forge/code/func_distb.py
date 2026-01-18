import sys
import types
import collections
import io
from opcode import *
from opcode import (
def distb(tb=None, *, file=None, show_caches=False, adaptive=False):
    """Disassemble a traceback (default: last traceback)."""
    if tb is None:
        try:
            tb = sys.last_traceback
        except AttributeError:
            raise RuntimeError('no last traceback to disassemble') from None
        while tb.tb_next:
            tb = tb.tb_next
    disassemble(tb.tb_frame.f_code, tb.tb_lasti, file=file, show_caches=show_caches, adaptive=adaptive)