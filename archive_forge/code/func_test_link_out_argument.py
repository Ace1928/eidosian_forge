import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn._loss.link import (
@pytest.mark.parametrize('link', LINK_FUNCTIONS)
def test_link_out_argument(link):
    rng = np.random.RandomState(42)
    link = link()
    n_samples, n_classes = (100, None)
    if link.is_multiclass:
        n_classes = 10
        raw_prediction = rng.normal(loc=0, scale=10, size=(n_samples, n_classes))
        if isinstance(link, MultinomialLogit):
            raw_prediction = link.symmetrize_raw_prediction(raw_prediction)
    else:
        raw_prediction = rng.uniform(low=-10, high=10, size=n_samples)
    y_pred = link.inverse(raw_prediction, out=None)
    out = np.empty_like(raw_prediction)
    y_pred_2 = link.inverse(raw_prediction, out=out)
    assert_allclose(y_pred, out)
    assert_array_equal(out, y_pred_2)
    assert np.shares_memory(out, y_pred_2)
    out = np.empty_like(y_pred)
    raw_prediction_2 = link.link(y_pred, out=out)
    assert_allclose(raw_prediction, out)
    assert_array_equal(out, raw_prediction_2)
    assert np.shares_memory(out, raw_prediction_2)