from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Union
import pytest
from xarray import DataArray, Dataset, Variable
@parametrize_dim
def test_dataset_variable_hashable_names(dim: DimT) -> None:
    Dataset({dim: ('x', [1, 2, 3])})