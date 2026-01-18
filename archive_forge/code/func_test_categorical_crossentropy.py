import numpy
import pytest
from thinc import registry
from thinc.api import (
@pytest.mark.parametrize('guesses, labels', [(guesses1, labels1), (guesses1, labels1_full)])
def test_categorical_crossentropy(guesses, labels):
    d_scores = CategoricalCrossentropy(normalize=True).get_grad(guesses, labels)
    assert d_scores.shape == guesses.shape
    assert d_scores[1][0] == pytest.approx(0.1, eps)
    assert d_scores[1][1] == pytest.approx(-0.1, eps)
    assert d_scores[2][0] == pytest.approx(0, eps)
    assert d_scores[2][1] == pytest.approx(0.25, eps)
    assert d_scores[2][2] == pytest.approx(0.25, eps)
    assert d_scores[3][0] == pytest.approx(0, eps)
    assert d_scores[3][1] == pytest.approx(0, eps)
    assert d_scores[3][2] == pytest.approx(-0.25, eps)
    loss = CategoricalCrossentropy(normalize=True).get_loss(guesses, labels)
    assert loss == pytest.approx(0.239375, eps)