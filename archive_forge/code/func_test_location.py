from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_location():
    """Make sure ``location()`` does what it claims."""
    t = TestTerminal(stream=StringIO(), force_styling=True)
    with t.location(3, 4):
        t.stream.write(u'hi')
    eq_(t.stream.getvalue(), unicode_cap('sc') + unicode_parm('cup', 4, 3) + u'hi' + unicode_cap('rc'))