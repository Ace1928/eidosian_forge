import logging
import unittest
import multiprocessing as mp
from functools import partial
import numpy as np
from gensim.matutils import argsort
from gensim.models.coherencemodel import CoherenceModel, BOOLEAN_DOCUMENT_BASED
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import get_tmpfile, common_texts, common_dictionary, common_corpus
def testProcesses(self):
    get_model = partial(CoherenceModel, topics=self.topics1, corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass')
    model, used_cpus = (get_model(), mp.cpu_count() - 1)
    self.assertEqual(model.processes, used_cpus)
    for p in range(-2, 1):
        self.assertEqual(get_model(processes=p).processes, used_cpus)
    for p in range(1, 4):
        self.assertEqual(get_model(processes=p).processes, p)