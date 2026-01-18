from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_capability_with_forced_tty():
    """If we force styling, capabilities had better not (generally) be
    empty."""
    t = TestTerminal(stream=StringIO(), force_styling=True)
    eq_(t.save, unicode_cap('sc'))