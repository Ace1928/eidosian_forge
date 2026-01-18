import numpy
import pytest
from thinc import registry
from thinc.api import (
def test_crossentropy_incorrect_scores_targets():
    labels = numpy.asarray([2])
    guesses_neg = numpy.asarray([[-0.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match='Cannot calculate.*guesses'):
        CategoricalCrossentropy(normalize=True).get_grad(guesses_neg, labels)
    guesses_larger_than_one = numpy.asarray([[1.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match='Cannot calculate.*guesses'):
        CategoricalCrossentropy(normalize=True).get_grad(guesses_larger_than_one, labels)
    guesses_ok = numpy.asarray([[0.1, 0.4, 0.5]])
    targets_neg = numpy.asarray([[-0.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match='Cannot calculate.*truth'):
        CategoricalCrossentropy(normalize=True).get_grad(guesses_ok, targets_neg)
    targets_larger_than_one = numpy.asarray([[2.0, 0.5, 0.6]])
    with pytest.raises(ValueError, match='Cannot calculate.*truth'):
        CategoricalCrossentropy(normalize=True).get_grad(guesses_ok, targets_larger_than_one)