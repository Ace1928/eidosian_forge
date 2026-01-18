from functools import partial
import pytest
from sklearn.datasets.tests.test_common import (
@pytest.mark.parametrize('as_frame', [True, False])
@pytest.mark.parametrize('subset, n_samples, n_features', [(None, 494021, 41), ('SA', 100655, 41), ('SF', 73237, 4), ('http', 58725, 3), ('smtp', 9571, 3)])
def test_fetch_kddcup99_percent10(fetch_kddcup99_fxt, as_frame, subset, n_samples, n_features):
    data = fetch_kddcup99_fxt(subset=subset, as_frame=as_frame)
    assert data.data.shape == (n_samples, n_features)
    assert data.target.shape == (n_samples,)
    if as_frame:
        assert data.frame.shape == (n_samples, n_features + 1)
    assert data.DESCR.startswith('.. _kddcup99_dataset:')