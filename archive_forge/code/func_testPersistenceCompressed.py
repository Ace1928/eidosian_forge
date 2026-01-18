import logging
import unittest
import multiprocessing as mp
from functools import partial
import numpy as np
from gensim.matutils import argsort
from gensim.models.coherencemodel import CoherenceModel, BOOLEAN_DOCUMENT_BASED
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import get_tmpfile, common_texts, common_dictionary, common_corpus
def testPersistenceCompressed(self):
    fname = get_tmpfile('gensim_models_coherence.tst.gz')
    model = CoherenceModel(topics=self.topics1, corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass')
    model.save(fname)
    model2 = CoherenceModel.load(fname)
    self.assertTrue(model.get_coherence() == model2.get_coherence())