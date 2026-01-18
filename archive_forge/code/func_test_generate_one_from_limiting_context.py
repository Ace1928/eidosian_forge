import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_generate_one_from_limiting_context(mle_trigram_model):
    assert mle_trigram_model.generate(text_seed=['c']) == 'd'
    assert mle_trigram_model.generate(text_seed=['b', 'c']) == 'd'
    assert mle_trigram_model.generate(text_seed=['a', 'c']) == 'd'