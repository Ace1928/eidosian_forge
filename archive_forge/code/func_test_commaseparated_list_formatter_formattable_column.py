import argparse
import io
import unittest
from unittest import mock
from cliff.formatters import commaseparated
from cliff.tests import test_columns
def test_commaseparated_list_formatter_formattable_column(self):
    sf = commaseparated.CSVLister()
    c = ('a', 'b', 'c')
    d1 = ('A', 'B', test_columns.FauxColumn(['the', 'value']))
    data = [d1]
    expected = "a,b,c\nA,B,['the'\\, 'value']\n"
    output = io.StringIO()
    parsed_args = mock.Mock()
    parsed_args.quote_mode = 'none'
    sf.emit_list(c, data, output, parsed_args)
    actual = output.getvalue()
    self.assertEqual(expected, actual)