import logging
import unittest
import multiprocessing as mp
from functools import partial
import numpy as np
from gensim.matutils import argsort
from gensim.models.coherencemodel import CoherenceModel, BOOLEAN_DOCUMENT_BASED
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import get_tmpfile, common_texts, common_dictionary, common_corpus
def testAccumulatorCachingTopicSubsets(self):
    kwargs = dict(corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass')
    cm1 = CoherenceModel(topics=self.topics1, **kwargs)
    cm1.estimate_probabilities()
    accumulator = cm1._accumulator
    self.assertIsNotNone(accumulator)
    cm1.topics = [t[:2] for t in self.topics1]
    self.assertEqual(accumulator, cm1._accumulator)
    cm1.topics = self.topics1
    self.assertEqual(accumulator, cm1._accumulator)