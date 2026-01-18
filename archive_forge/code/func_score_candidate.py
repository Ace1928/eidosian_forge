import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def score_candidate(self, word_a, word_b, in_between):
    phrase = '_'.join([word_a] + in_between + [word_b])
    score = self.scores.get(phrase, -1)
    if score > self.threshold:
        return (phrase, score)
    return (None, None)