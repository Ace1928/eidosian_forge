from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
@pytest.mark.parametrize('display_width', [40, 80, 100])
@pytest.mark.parametrize('periods', [2, 3, 4, 100, 101])
def test_cftimeindex_repr_formatting_width(periods, display_width):
    """Test that cftimeindex is sensitive to OPTIONS['display_width']."""
    index = xr.cftime_range(start='2000', periods=periods)
    len_intro_str = len('CFTimeIndex(')
    with xr.set_options(display_width=display_width):
        repr_str = index.__repr__()
        splitted = repr_str.split('\n')
        for i, s in enumerate(splitted):
            assert len(s) <= display_width, f'{len(s)} {s} {display_width}'
            if i > 0:
                assert s[:len_intro_str] == ' ' * len_intro_str