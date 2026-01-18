import os
import pytest
import sys
import textwrap
import unittest
from contextlib import contextmanager
from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec
from IPython.core.completer import (
def recompute_unicode_ranges():
    """
    utility to recompute the largest unicode range without any characters

    use to recompute the gap in the global _UNICODE_RANGES of completer.py
    """
    import itertools
    import unicodedata
    valid = []
    for c in range(0, 1114111 + 1):
        try:
            unicodedata.name(chr(c))
        except ValueError:
            continue
        valid.append(c)

    def ranges(i):
        for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
            b = list(b)
            yield (b[0][1], b[-1][1])
    rg = list(ranges(valid))
    lens = []
    gap_lens = []
    pstart, pstop = (0, 0)
    for start, stop in rg:
        lens.append(stop - start)
        gap_lens.append((start - pstop, hex(pstop + 1), hex(start), f'{round((start - pstop) / 918000 * 100)}%'))
        pstart, pstop = (start, stop)
    return sorted(gap_lens)[-1]