import logging
import unittest
import os
import numpy as np
import gensim
from gensim.test.utils import get_tmpfile
def test_lsi_model(self):
    corpus = BigCorpus(num_docs=50000)
    tmpf = get_tmpfile('gensim_big.tst')
    model = gensim.models.LsiModel(corpus, num_topics=500, id2word=corpus.dictionary)
    model.save(tmpf)
    del model
    gensim.models.LsiModel.load(tmpf)