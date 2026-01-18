from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_init_descriptor_always_initted():
    """We should be able to get a height and width even on no-tty Terminals."""
    t = Terminal(stream=StringIO())
    eq_(type(t.height), int)