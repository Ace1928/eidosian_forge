import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_generate_cycle(mle_trigram_model):
    more_training_text = [padded_everygrams(mle_trigram_model.order, list('bdbdbd'))]
    mle_trigram_model.fit(more_training_text)
    assert mle_trigram_model.generate(7, text_seed=('b', 'd'), random_seed=5) == ['b', 'd', 'b', 'd', 'b', 'd', '</s>']