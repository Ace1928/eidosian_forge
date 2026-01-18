import argparse
import os
import textwrap
from io import StringIO
from unittest import mock
from cliff.formatters import table
from cliff.tests import base
from cliff.tests import test_columns
@mock.patch.dict(os.environ, {'CLIFF_MAX_TERM_WIDTH': '42'})
def test_env_maxwidth_args_tiny(self):
    self.assertEqual(self._expected_mv[40], _table_tester_helper(self._col_names, self._col_data, extra_args=args(40)))