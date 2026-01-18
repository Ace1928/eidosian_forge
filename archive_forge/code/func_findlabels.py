import sys
import types
import collections
import io
from opcode import *
from opcode import (
def findlabels(code):
    """Detect all offsets in a byte code which are jump targets.

    Return the list of offsets.

    """
    labels = []
    for offset, op, arg in _unpack_opargs(code):
        if arg is not None:
            if op in hasjrel:
                if _is_backward_jump(op):
                    arg = -arg
                label = offset + 2 + arg * 2
            elif op in hasjabs:
                label = arg * 2
            else:
                continue
            if label not in labels:
                labels.append(label)
    return labels