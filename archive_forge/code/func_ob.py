from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def ob(*order_by):
    ws = WindowSpec('', partition_keys=[], order_by=OrderBySpec(*order_by), windowframe=make_windowframe_spec(''))
    return WindowFunctionSpec('row_number', False, False, ws)