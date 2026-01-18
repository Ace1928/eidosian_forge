from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_callable_numeric_colors():
    """``color(n)`` should return a formatting wrapper."""
    t = TestTerminal()
    eq_(t.color(5)('smoo'), t.magenta + 'smoo' + t.normal)
    eq_(t.color(5)('smoo'), t.color(5) + 'smoo' + t.normal)
    eq_(t.on_color(2)('smoo'), t.on_green + 'smoo' + t.normal)
    eq_(t.on_color(2)('smoo'), t.on_color(2) + 'smoo' + t.normal)