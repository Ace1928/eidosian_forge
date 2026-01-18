import logging
import unittest
import multiprocessing as mp
from functools import partial
import numpy as np
from gensim.matutils import argsort
from gensim.models.coherencemodel import CoherenceModel, BOOLEAN_DOCUMENT_BASED
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import get_tmpfile, common_texts, common_dictionary, common_corpus
def testAccumulatorCachingWithModelSetting(self):
    kwargs = dict(corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass')
    cm1 = CoherenceModel(topics=self.topics1, **kwargs)
    cm1.estimate_probabilities()
    self.assertIsNotNone(cm1._accumulator)
    cm1.model = self.ldamodel
    topics = []
    for topic in self.ldamodel.state.get_lambda():
        bestn = argsort(topic, topn=cm1.topn, reverse=True)
        topics.append(bestn)
    self.assertTrue(np.array_equal(topics, cm1.topics))
    self.assertIsNone(cm1._accumulator)