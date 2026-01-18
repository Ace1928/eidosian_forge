import collections
import inspect
import socket
import sys
import tempfile
import unittest
from typing import List, Tuple
from itertools import islice
from pathlib import Path
from unittest import mock
from bpython import config, repl, cli, autocomplete
from bpython.line import LinePart
from bpython.test import (
def test_function_expressions(self):
    te = self.assertTupleEqual
    fa = lambda line: repl.Repl._funcname_and_argnum(line)
    for line, (func, argnum) in [('spam(', ('spam', 0)), ('spam((), ', ('spam', 1)), ('spam.eggs((), ', ('spam.eggs', 1)), ('spam[abc].eggs((), ', ('spam[abc].eggs', 1)), ('spam[0].eggs((), ', ('spam[0].eggs', 1)), ('spam[a + b]eggs((), ', ('spam[a + b]eggs', 1)), ('spam().eggs((), ', ('spam().eggs', 1)), ('spam(1, 2).eggs((), ', ('spam(1, 2).eggs', 1)), ('spam(1, f(1)).eggs((), ', ('spam(1, f(1)).eggs', 1)), ('[0].eggs((), ', ('[0].eggs', 1)), ('[0][0]((), {}).eggs((), ', ('[0][0]((), {}).eggs', 1)), ('a + spam[0].eggs((), ', ('spam[0].eggs', 1)), ('spam(', ('spam', 0)), ('spam(map([]', ('map', 0)), ('spam((), ', ('spam', 1))]:
        te(fa(line), (func, argnum))