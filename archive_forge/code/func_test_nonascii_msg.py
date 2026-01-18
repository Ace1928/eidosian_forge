import io
import os.path
import platform
import re
import sys
import traceback
import unittest
from textwrap import dedent
from tempfile import TemporaryDirectory
from IPython.core.ultratb import ColorTB, VerboseTB
from IPython.testing import tools as tt
from IPython.testing.decorators import onlyif_unicode_paths, skip_without
from IPython.utils.syspathcontext import prepended_to_syspath
import sys
def test_nonascii_msg(self):
    cell = u"raise Exception('é')"
    expected = u"Exception('é')"
    ip.run_cell('%xmode plain')
    with tt.AssertPrints(expected):
        ip.run_cell(cell)
    ip.run_cell('%xmode verbose')
    with tt.AssertPrints(expected):
        ip.run_cell(cell)
    ip.run_cell('%xmode context')
    with tt.AssertPrints(expected):
        ip.run_cell(cell)
    ip.run_cell('%xmode minimal')
    with tt.AssertPrints(u'Exception: é'):
        ip.run_cell(cell)
    ip.run_cell('%xmode context')