from __future__ import annotations
import sys
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray
import xarray as xr
from xarray.core import formatting
from xarray.tests import requires_cftime, requires_dask, requires_netCDF4
def test_attribute_repr(self) -> None:
    short = formatting.summarize_attr('key', 'Short string')
    long = formatting.summarize_attr('key', 100 * 'Very long string ')
    newlines = formatting.summarize_attr('key', '\n\n\n')
    tabs = formatting.summarize_attr('key', '\t\t\t')
    assert short == '    key: Short string'
    assert len(long) <= 80
    assert long.endswith('...')
    assert '\n' not in newlines
    assert '\t' not in tabs