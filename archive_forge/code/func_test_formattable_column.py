import argparse
import io
from unittest import mock
from cliff.formatters import shell
from cliff.tests import base
from cliff.tests import test_columns
def test_formattable_column(self):
    sf = shell.ShellFormatter()
    c = ('a', 'b', 'c')
    d = ('A', 'B', test_columns.FauxColumn(['the', 'value']))
    expected = '\n'.join(['a="A"', 'b="B"', 'c="[\'the\', \'value\']"\n'])
    output = io.StringIO()
    args = mock.Mock()
    args.variables = ['a', 'b', 'c']
    args.prefix = ''
    sf.emit_one(c, d, output, args)
    actual = output.getvalue()
    self.assertEqual(expected, actual)