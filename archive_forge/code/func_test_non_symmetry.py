from functools import partial
from itertools import chain
import numpy as np
import pytest
from sklearn.metrics.cluster import (
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('metric_name, y1, y2', [(name, y1, y2) for name in NON_SYMMETRIC_METRICS])
def test_non_symmetry(metric_name, y1, y2):
    metric = SUPERVISED_METRICS[metric_name]
    assert metric(y1, y2) != pytest.approx(metric(y2, y1))