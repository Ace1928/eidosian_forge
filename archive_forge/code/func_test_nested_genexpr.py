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
def test_nested_genexpr(self):
    code = dedent('            class SpecificException(Exception):\n                pass\n\n            def foo(x):\n                raise SpecificException("Success!")\n\n            sum(sum(foo(x) for _ in [0]) for x in [0])\n            ')
    with tt.AssertPrints('SpecificException: Success!', suppress=False):
        ip.run_cell(code)