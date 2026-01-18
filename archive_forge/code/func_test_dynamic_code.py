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
@skip_without('pandas')
def test_dynamic_code():
    code = '\n    import pandas\n    df = pandas.DataFrame([])\n\n    # Important: only fails inside of an "exec" call:\n    exec("df.foobarbaz()")\n    '
    with tt.AssertPrints('Could not get source'):
        ip.run_cell(code)