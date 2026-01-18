import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_find_phrases(self):
    """Test Phrases bigram export phrases."""
    bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words, delimiter=' ')
    seen_bigrams = set(bigram.find_phrases(self.sentences).keys())
    assert seen_bigrams == set(['human interface', 'graph of trees', 'data and graph', 'lack of interest'])