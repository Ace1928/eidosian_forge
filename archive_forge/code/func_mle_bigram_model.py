import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.fixture
def mle_bigram_model(vocabulary, bigram_training_data):
    model = MLE(2, vocabulary=vocabulary)
    model.fit(bigram_training_data)
    return model