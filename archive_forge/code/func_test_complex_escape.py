import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_complex_escape(self):
    s = '\\r \\\n'
    e = '\\\\r \\\\\\n'
    self.assertEqual(export_pot._escape(s), e)