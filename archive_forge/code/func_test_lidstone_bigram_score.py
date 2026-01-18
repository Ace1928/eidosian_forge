import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.mark.parametrize('word, context, expected_score', [('d', ['c'], 1.1 / 1.8), ('a', None, 2.1 / 14.8), ('z', None, 0.1 / 14.8), ('y', None, 3.1 / 14.8)])
def test_lidstone_bigram_score(lidstone_bigram_model, word, context, expected_score):
    assert pytest.approx(lidstone_bigram_model.score(word, context), 0.0001) == expected_score