import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.mark.parametrize('word, context, expected_score', [('d', ['c'], 1.1 / 1.8), ('e', ['c'], 0.1 / 1.8), ('d', ['b', 'c'], 1.1 / 1.8), ('e', ['b', 'c'], 0.1 / 1.8)])
def test_lidstone_trigram_score(lidstone_trigram_model, word, context, expected_score):
    assert pytest.approx(lidstone_trigram_model.score(word, context), 0.0001) == expected_score