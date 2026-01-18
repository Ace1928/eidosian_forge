import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_bigram_construction_from_array(self):
    """Test Phrases bigram construction building when corpus is a numpy array."""
    bigram1_seen = False
    bigram2_seen = False
    for s in self.bigram[np.array(self.sentences, dtype=object)]:
        if not bigram1_seen and self.bigram1 in s:
            bigram1_seen = True
        if not bigram2_seen and self.bigram2 in s:
            bigram2_seen = True
        if bigram1_seen and bigram2_seen:
            break
    self.assertTrue(bigram1_seen and bigram2_seen)