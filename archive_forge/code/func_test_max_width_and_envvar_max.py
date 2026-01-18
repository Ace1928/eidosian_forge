import argparse
import os
import textwrap
from io import StringIO
from unittest import mock
from cliff.formatters import table
from cliff.tests import base
from cliff.tests import test_columns
@mock.patch('cliff.utils.terminal_width')
@mock.patch.dict(os.environ, {'CLIFF_MAX_TERM_WIDTH': '666'})
def test_max_width_and_envvar_max(self, tw):
    tw.return_value = 80
    self.assertEqual(self._expected_mv[80], _table_tester_helper(self._col_names, self._col_data))
    tw.return_value = 50
    self.assertEqual(self._expected_mv[80], _table_tester_helper(self._col_names, self._col_data))
    tw.return_value = 45
    self.assertEqual(self._expected_mv[80], _table_tester_helper(self._col_names, self._col_data))
    tw.return_value = 40
    self.assertEqual(self._expected_mv[80], _table_tester_helper(self._col_names, self._col_data))
    tw.return_value = 10
    self.assertEqual(self._expected_mv[80], _table_tester_helper(self._col_names, self._col_data))