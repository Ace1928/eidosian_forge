from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_force_styling_none():
    """If ``force_styling=None`` is passed to the constructor, don't ever do
    styling."""
    t = TestTerminal(force_styling=None)
    eq_(t.save, '')