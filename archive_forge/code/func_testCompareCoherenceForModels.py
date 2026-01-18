import logging
import unittest
import multiprocessing as mp
from functools import partial
import numpy as np
from gensim.matutils import argsort
from gensim.models.coherencemodel import CoherenceModel, BOOLEAN_DOCUMENT_BASED
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import get_tmpfile, common_texts, common_dictionary, common_corpus
def testCompareCoherenceForModels(self):
    models = [self.ldamodel, self.ldamodel]
    cm = CoherenceModel.for_models(models, dictionary=self.dictionary, texts=self.texts, coherence='c_v')
    self.assertIsNotNone(cm._accumulator)
    for model in models:
        cm.model = model
        self.assertIsNotNone(cm._accumulator)
    (coherence_topics1, coherence1), (coherence_topics2, coherence2) = cm.compare_models(models)
    self.assertAlmostEqual(np.mean(coherence_topics1), coherence1, 4)
    self.assertAlmostEqual(np.mean(coherence_topics2), coherence2, 4)
    self.assertAlmostEqual(coherence1, coherence2, places=4)