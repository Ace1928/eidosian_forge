from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_formatting_functions_without_tty():
    """Test crazy-ass formatting wrappers when there's no tty."""
    t = TestTerminal(stream=StringIO())
    eq_(t.bold(u'hi'), u'hi')
    eq_(t.green('hi'), u'hi')
    eq_(t.bold_green(u'boö'), u'boö')
    eq_(t.bold_underline_green_on_red('loo'), u'loo')
    eq_(t.on_bright_red_bold_bright_green_underline('meh'), u'meh')