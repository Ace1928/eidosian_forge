from io import StringIO
import yaml
from unittest import mock
from cliff.formatters import yaml_format
from cliff.tests import base
from cliff.tests import test_columns
def test_format_one(self):
    sf = yaml_format.YAMLFormatter()
    c = ('a', 'b', 'c', 'd')
    d = ('A', 'B', 'C', '"escape me"')
    expected = {'a': 'A', 'b': 'B', 'c': 'C', 'd': '"escape me"'}
    output = StringIO()
    args = mock.Mock()
    sf.emit_one(c, d, output, args)
    actual = yaml.safe_load(output.getvalue())
    self.assertEqual(expected, actual)