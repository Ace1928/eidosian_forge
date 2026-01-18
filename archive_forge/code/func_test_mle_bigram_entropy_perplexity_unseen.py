import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_mle_bigram_entropy_perplexity_unseen(mle_bigram_model):
    untrained = [('<s>', 'a'), ('a', 'c'), ('c', 'd'), ('d', '</s>')]
    assert math.isinf(mle_bigram_model.entropy(untrained))
    assert math.isinf(mle_bigram_model.perplexity(untrained))