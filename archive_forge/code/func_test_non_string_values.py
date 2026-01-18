import argparse
import io
from unittest import mock
from cliff.formatters import shell
from cliff.tests import base
from cliff.tests import test_columns
def test_non_string_values(self):
    sf = shell.ShellFormatter()
    c = ('a', 'b', 'c', 'd', 'e')
    d = (True, False, 100, '"esc"', str('"esc"'))
    expected = 'a="True"\nb="False"\nc="100"\nd="\\"esc\\""\ne="\\"esc\\""\n'
    output = io.StringIO()
    args = mock.Mock()
    args.variables = ['a', 'b', 'c', 'd', 'e']
    args.prefix = ''
    sf.emit_one(c, d, output, args)
    actual = output.getvalue()
    self.assertEqual(expected, actual)