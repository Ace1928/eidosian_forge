import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_generate_with_text_seed(mle_trigram_model):
    assert mle_trigram_model.generate(5, text_seed=('<s>', 'e'), random_seed=3) == ['<UNK>', 'a', 'd', 'b', '<UNK>']