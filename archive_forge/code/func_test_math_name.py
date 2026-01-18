from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_math_name(self) -> None:
    a = self.dv
    assert (+a).name == 'foo'
    assert (a + 0).name == 'foo'
    assert (a + a.rename(None)).name is None
    assert (a + a.rename('bar')).name is None
    assert (a + a).name == 'foo'
    assert (+a['x']).name == 'x'
    assert (a['x'] + 0).name == 'x'
    assert (a + a['x']).name is None