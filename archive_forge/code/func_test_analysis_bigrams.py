import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_analysis_bigrams(self):
    scores = {'simple_sentence': 2, 'sentence_many': 2, 'many_possible': 2, 'possible_bigrams': 2}
    sentence = ['simple', 'sentence', 'many', 'possible', 'bigrams']
    result = self.AnalysisTester(scores, threshold=1)[sentence]
    self.assertEqual(result, ['simple_sentence', 'many_possible', 'bigrams'])
    sentence = ['some', 'simple', 'sentence', 'many', 'bigrams']
    result = self.AnalysisTester(scores, threshold=1)[sentence]
    self.assertEqual(result, ['some', 'simple_sentence', 'many', 'bigrams'])
    sentence = ['some', 'unrelated', 'simple', 'words']
    result = self.AnalysisTester(scores, threshold=1)[sentence]
    self.assertEqual(result, sentence)