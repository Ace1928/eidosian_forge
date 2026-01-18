import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_compatibility(self):
    phrases = Phrases.load(datapath('phrases-3.6.0.model'))
    phraser = FrozenPhrases.load(datapath('phraser-3.6.0.model'))
    test_sentences = ['trees', 'graph', 'minors']
    self.assertEqual(phrases[test_sentences], ['trees', 'graph_minors'])
    self.assertEqual(phraser[test_sentences], ['trees', 'graph_minors'])