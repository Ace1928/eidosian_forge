from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_capability_without_tty():
    """Assert capability templates are '' when stream is not a tty."""
    t = TestTerminal(stream=StringIO())
    eq_(t.save, u'')
    eq_(t.red, u'')