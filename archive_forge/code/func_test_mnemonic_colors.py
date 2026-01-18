from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_mnemonic_colors():
    """Make sure color shortcuts work."""

    def color(num):
        return unicode_parm('setaf', num)

    def on_color(num):
        return unicode_parm('setab', num)
    t = TestTerminal()
    eq_(t.white, color(7))
    eq_(t.green, color(2))
    eq_(t.on_black, on_color(0))
    eq_(t.on_green, on_color(2))
    eq_(t.bright_black, color(8))
    eq_(t.bright_green, color(10))
    eq_(t.on_bright_black, on_color(8))
    eq_(t.on_bright_green, on_color(10))