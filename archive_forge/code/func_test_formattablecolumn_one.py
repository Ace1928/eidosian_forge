import io
import json
from unittest import mock
from cliff.formatters import json_format
from cliff.tests import base
from cliff.tests import test_columns
def test_formattablecolumn_one(self):
    sf = json_format.JSONFormatter()
    c = ('a', 'b', 'c', 'd')
    d = ('A', 'B', 'C', test_columns.FauxColumn(['the', 'value']))
    expected = {'a': 'A', 'b': 'B', 'c': 'C', 'd': ['the', 'value']}
    args = mock.Mock()
    sf.add_argument_group(args)
    args.noindent = True
    output = io.StringIO()
    sf.emit_one(c, d, output, args)
    value = output.getvalue()
    print(len(value.splitlines()))
    self.assertEqual(1, len(value.splitlines()))
    actual = json.loads(value)
    self.assertEqual(expected, actual)