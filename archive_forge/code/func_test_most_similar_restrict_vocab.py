import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_most_similar_restrict_vocab(self):
    """Test most_similar returns handles restrict_vocab correctly."""
    expected = set(self.vectors.index_to_key[:5])
    predicted = set((result[0] for result in self.vectors.most_similar('war', topn=5, restrict_vocab=5)))
    self.assertEqual(expected, predicted)