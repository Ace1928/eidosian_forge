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
def test_direct_cause_error(self):
    with tt.AssertPrints(['KeyError', 'NameError', 'direct cause']):
        ip.run_cell(self.DIRECT_CAUSE_ERROR_CODE)