import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_mle_bigram_entropy_perplexity_unigrams(mle_bigram_model):
    H = 3.0095
    perplexity = 8.0529
    text = [('<s>',), ('a',), ('c',), ('-',), ('d',), ('c',), ('</s>',)]
    assert pytest.approx(mle_bigram_model.entropy(text), 0.0001) == H
    assert pytest.approx(mle_bigram_model.perplexity(text), 0.0001) == perplexity