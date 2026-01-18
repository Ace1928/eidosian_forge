import argparse
import os
import textwrap
from io import StringIO
from unittest import mock
from cliff.formatters import table
from cliff.tests import base
from cliff.tests import test_columns
@mock.patch('cliff.utils.terminal_width')
def test_80(self, tw):
    tw.return_value = 80
    c = ('field_name', 'a_really_long_field_name')
    d = ('the value', 'a value significantly longer than the field')
    self.assertEqual(self.expected_80, _table_tester_helper(c, d))