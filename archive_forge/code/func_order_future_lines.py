from __future__ import print_function, absolute_import
import os
import tempfile
import unittest
import sys
import re
import warnings
import io
from textwrap import dedent
from future.utils import bind_method, PY26, PY3, PY2, PY27
from future.moves.subprocess import check_output, STDOUT, CalledProcessError
def order_future_lines(code):
    """
    Returns the code block with any ``__future__`` import lines sorted, and
    then any ``future`` import lines sorted, then any ``builtins`` import lines
    sorted.

    This only sorts the lines within the expected blocks.

    See test_order_future_lines() for an example.
    """
    lines = code.split('\n')
    uufuture_line_numbers = [i for i, line in enumerate(lines) if line.startswith('from __future__ import ')]
    future_line_numbers = [i for i, line in enumerate(lines) if line.startswith('from future') or line.startswith('from past')]
    builtins_line_numbers = [i for i, line in enumerate(lines) if line.startswith('from builtins')]
    assert code.lstrip() == code, 'internal usage error: dedent the code before calling order_future_lines()'

    def mymax(numbers):
        return max(numbers) if len(numbers) > 0 else 0

    def mymin(numbers):
        return min(numbers) if len(numbers) > 0 else float('inf')
    assert mymax(uufuture_line_numbers) <= mymin(future_line_numbers), 'the __future__ and future imports are out of order'
    uul = sorted([lines[i] for i in uufuture_line_numbers])
    sorted_uufuture_lines = dict(zip(uufuture_line_numbers, uul))
    fl = sorted([lines[i] for i in future_line_numbers])
    sorted_future_lines = dict(zip(future_line_numbers, fl))
    bl = sorted([lines[i] for i in builtins_line_numbers])
    sorted_builtins_lines = dict(zip(builtins_line_numbers, bl))
    new_lines = []
    for i in range(len(lines)):
        if i in uufuture_line_numbers:
            new_lines.append(sorted_uufuture_lines[i])
        elif i in future_line_numbers:
            new_lines.append(sorted_future_lines[i])
        elif i in builtins_line_numbers:
            new_lines.append(sorted_builtins_lines[i])
        else:
            new_lines.append(lines[i])
    return '\n'.join(new_lines)