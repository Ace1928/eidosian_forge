from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
def test_cupy_import() -> None:
    """Check the import worked."""
    assert cp