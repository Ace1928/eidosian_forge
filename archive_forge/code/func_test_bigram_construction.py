import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_bigram_construction(self):
    """Test Phrases bigram construction."""
    bigram1_seen = False
    bigram2_seen = False
    for sentence in self.bigram[self.sentences]:
        if not bigram1_seen and self.bigram1 in sentence:
            bigram1_seen = True
        if not bigram2_seen and self.bigram2 in sentence:
            bigram2_seen = True
        if bigram1_seen and bigram2_seen:
            break
    self.assertTrue(bigram1_seen and bigram2_seen)
    self.assertTrue(self.bigram1 in self.bigram[self.sentences[1]])
    self.assertTrue(self.bigram1 in self.bigram[self.sentences[4]])
    self.assertTrue(self.bigram2 in self.bigram[self.sentences[-2]])
    self.assertTrue(self.bigram2 in self.bigram[self.sentences[-1]])
    self.assertTrue(self.bigram3 in self.bigram[self.sentences[-1]])