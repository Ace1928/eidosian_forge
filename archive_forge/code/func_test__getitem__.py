import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test__getitem__(self):
    """Test Phrases[sentences] with a single sentence."""
    bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words)
    test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
    phrased_sentence = next(bigram[test_sentences].__iter__())
    assert phrased_sentence == ['data_and_graph', 'survey', 'for', 'human_interface']