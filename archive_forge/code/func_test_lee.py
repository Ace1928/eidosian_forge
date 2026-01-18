from __future__ import with_statement
import logging
import unittest
from functools import partial
import numpy as np
from gensim import corpora, models, utils, matutils
from gensim.parsing.preprocessing import preprocess_documents, preprocess_string, DEFAULT_FILTERS
from gensim.test.utils import datapath
def test_lee(self):
    """correlation with human data > 0.6
        (this is the value which was achieved in the original paper)
        """
    global bg_corpus, corpus
    dictionary = corpora.Dictionary(bg_corpus)
    bg_corpus = [dictionary.doc2bow(text) for text in bg_corpus]
    corpus = [dictionary.doc2bow(text) for text in corpus]
    log_ent = models.LogEntropyModel(bg_corpus)
    bg_corpus_ent = log_ent[bg_corpus]
    lsi = models.LsiModel(bg_corpus_ent, id2word=dictionary, num_topics=200)
    corpus_lsi = lsi[log_ent[corpus]]
    res = np.zeros((len(corpus), len(corpus)))
    for i, par1 in enumerate(corpus_lsi):
        for j, par2 in enumerate(corpus_lsi):
            res[i, j] = matutils.cossim(par1, par2)
    flat = res[np.triu_indices(len(corpus), 1)]
    cor = np.corrcoef(flat, human_sim_vector)[0, 1]
    logging.info('LSI correlation coefficient is %s', cor)
    self.assertTrue(cor > 0.6)