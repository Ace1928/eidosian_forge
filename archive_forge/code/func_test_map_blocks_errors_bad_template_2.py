from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
def test_map_blocks_errors_bad_template_2(map_ds):
    with pytest.raises(ValueError, match="unexpected data variables {'xyz'}"):
        xr.map_blocks(lambda x: x.assign(xyz=1), map_ds, template=map_ds).compute()