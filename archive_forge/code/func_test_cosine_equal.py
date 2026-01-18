import numpy
import pytest
from thinc import registry
from thinc.api import (
def test_cosine_equal():
    vec1 = numpy.asarray([[1, 2], [8, 9], [3, 3]])
    vec2 = numpy.asarray([[1, 2], [80, 90], [300, 300]])
    d_vec1 = CosineDistance().get_grad(vec1, vec2)
    assert d_vec1.shape == vec1.shape
    numpy.testing.assert_allclose(d_vec1, numpy.zeros(d_vec1.shape), rtol=eps, atol=eps)
    loss_not_normalized = CosineDistance(normalize=False).get_loss(vec1, vec2)
    assert loss_not_normalized == pytest.approx(0, eps)
    loss_normalized = CosineDistance(normalize=True).get_loss(vec1, vec2)
    assert loss_normalized == pytest.approx(0, eps)