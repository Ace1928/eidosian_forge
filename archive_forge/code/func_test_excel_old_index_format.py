from __future__ import annotations
from datetime import (
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_excel_old_index_format(self, read_ext):
    filename = 'test_index_name_pre17' + read_ext
    data = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan], ['R0C0', 'R0C1', 'R0C2', 'R0C3', 'R0C4'], ['R1C0', 'R1C1', 'R1C2', 'R1C3', 'R1C4'], ['R2C0', 'R2C1', 'R2C2', 'R2C3', 'R2C4'], ['R3C0', 'R3C1', 'R3C2', 'R3C3', 'R3C4'], ['R4C0', 'R4C1', 'R4C2', 'R4C3', 'R4C4']], dtype=object)
    columns = ['C_l0_g0', 'C_l0_g1', 'C_l0_g2', 'C_l0_g3', 'C_l0_g4']
    mi = MultiIndex(levels=[['R0', 'R_l0_g0', 'R_l0_g1', 'R_l0_g2', 'R_l0_g3', 'R_l0_g4'], ['R1', 'R_l1_g0', 'R_l1_g1', 'R_l1_g2', 'R_l1_g3', 'R_l1_g4']], codes=[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], names=[None, None])
    si = Index(['R0', 'R_l0_g0', 'R_l0_g1', 'R_l0_g2', 'R_l0_g3', 'R_l0_g4'], name=None)
    expected = DataFrame(data, index=si, columns=columns)
    actual = pd.read_excel(filename, sheet_name='single_names', index_col=0)
    tm.assert_frame_equal(actual, expected)
    expected.index = mi
    actual = pd.read_excel(filename, sheet_name='multi_names', index_col=[0, 1])
    tm.assert_frame_equal(actual, expected)
    data = np.array([['R0C0', 'R0C1', 'R0C2', 'R0C3', 'R0C4'], ['R1C0', 'R1C1', 'R1C2', 'R1C3', 'R1C4'], ['R2C0', 'R2C1', 'R2C2', 'R2C3', 'R2C4'], ['R3C0', 'R3C1', 'R3C2', 'R3C3', 'R3C4'], ['R4C0', 'R4C1', 'R4C2', 'R4C3', 'R4C4']])
    columns = ['C_l0_g0', 'C_l0_g1', 'C_l0_g2', 'C_l0_g3', 'C_l0_g4']
    mi = MultiIndex(levels=[['R_l0_g0', 'R_l0_g1', 'R_l0_g2', 'R_l0_g3', 'R_l0_g4'], ['R_l1_g0', 'R_l1_g1', 'R_l1_g2', 'R_l1_g3', 'R_l1_g4']], codes=[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], names=[None, None])
    si = Index(['R_l0_g0', 'R_l0_g1', 'R_l0_g2', 'R_l0_g3', 'R_l0_g4'], name=None)
    expected = DataFrame(data, index=si, columns=columns)
    actual = pd.read_excel(filename, sheet_name='single_no_names', index_col=0)
    tm.assert_frame_equal(actual, expected)
    expected.index = mi
    actual = pd.read_excel(filename, sheet_name='multi_no_names', index_col=[0, 1])
    tm.assert_frame_equal(actual, expected)