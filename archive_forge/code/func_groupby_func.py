import numpy as np
import pytest
from pandas import (
from pandas.core.groupby.base import (
@pytest.fixture(params=sorted(reduction_kernels) + sorted(transformation_kernels))
def groupby_func(request):
    """yields both aggregation and transformation functions."""
    return request.param