import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_duplicate(self):
    self.exporter.poentry('dummy', 1, 'spam')
    self.exporter.poentry('dummy', 2, 'spam', 'EGG')
    self.check_output('                #: dummy:1\n                msgid "spam"\n                msgstr ""\n\n                ')