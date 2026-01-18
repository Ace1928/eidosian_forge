import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_multiple_bigrams_single_entry(self):
    """Test a single entry produces multiple bigrams."""
    bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words, delimiter=' ')
    test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
    seen_bigrams = set(bigram.find_phrases(test_sentences).keys())
    assert seen_bigrams == set(['data and graph', 'human interface'])