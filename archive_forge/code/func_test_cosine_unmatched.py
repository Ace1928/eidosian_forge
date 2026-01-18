import numpy
import pytest
from thinc import registry
from thinc.api import (
def test_cosine_unmatched():
    vec1 = numpy.asarray([[1, 2, 3]])
    vec2 = numpy.asarray([[1, 2]])
    with pytest.raises(ValueError):
        CosineDistance().get_grad(vec1, vec2)