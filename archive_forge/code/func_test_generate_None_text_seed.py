import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_generate_None_text_seed(mle_trigram_model):
    with pytest.raises(TypeError):
        mle_trigram_model.generate(text_seed=(None,))
    assert mle_trigram_model.generate(text_seed=None, random_seed=3) == mle_trigram_model.generate(random_seed=3)