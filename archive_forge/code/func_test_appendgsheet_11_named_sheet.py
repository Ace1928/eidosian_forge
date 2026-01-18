from __future__ import absolute_import, print_function, division
import datetime
import os
import json
import time
import pytest
from petl.compat import text_type
from petl.io.gsheet import fromgsheet, togsheet, appendgsheet
from petl.test.helpers import ieq, get_env_vars_named
@found_gcp_credentials
def test_appendgsheet_11_named_sheet():
    table_list, expected = _get_testcase_for_append()
    _test_append_from_gsheet(table_list, expected, sheetname='petl_append')