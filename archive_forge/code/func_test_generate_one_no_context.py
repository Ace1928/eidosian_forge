import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_generate_one_no_context(mle_trigram_model):
    assert mle_trigram_model.generate(random_seed=3) == '<UNK>'