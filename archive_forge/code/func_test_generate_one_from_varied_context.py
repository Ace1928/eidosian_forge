import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_generate_one_from_varied_context(mle_trigram_model):
    assert mle_trigram_model.generate(text_seed=('a', '<s>'), random_seed=2) == 'a'