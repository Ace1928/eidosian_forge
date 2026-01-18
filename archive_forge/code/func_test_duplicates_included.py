import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_duplicates_included(self):
    exporter = export_pot._PotExporter(StringIO(), True)
    context = export_pot._ModuleContext('mod.py', 1)
    exporter.poentry_in_context(context, 'Common line.')
    context.lineno = 3
    exporter.poentry_in_context(context, 'Common line.')
    self.assertEqual(2, exporter.outf.getvalue().count('Common line.'))