import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_mle_bigram_logscore_for_zero_score(mle_bigram_model):
    assert math.isinf(mle_bigram_model.logscore('d', ['e']))