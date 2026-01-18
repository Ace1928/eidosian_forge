import logging
import unittest
import os
import numpy as np
import gensim
from gensim.test.utils import get_tmpfile
def test_lda_model(self):
    corpus = BigCorpus(num_docs=5000)
    tmpf = get_tmpfile('gensim_big.tst')
    model = gensim.models.LdaModel(corpus, num_topics=500, id2word=corpus.dictionary)
    model.save(tmpf)
    del model
    gensim.models.LdaModel.load(tmpf)