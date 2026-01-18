from __future__ import division  # always use floats
from __future__ import with_statement
import logging
import os
import unittest
from gensim import utils, corpora, models, similarities
from gensim.test.utils import datapath, get_tmpfile
def test_miislita_high_level(self):
    miislita = CorpusMiislita(datapath('miIslita.cor'))
    tfidf = models.TfidfModel(miislita, miislita.dictionary, normalize=False)
    index = similarities.SparseMatrixSimilarity(tfidf[miislita], num_features=len(miislita.dictionary))
    query = 'latent semantic indexing'
    vec_bow = miislita.dictionary.doc2bow(query.lower().split())
    vec_tfidf = tfidf[vec_bow]
    sims_tfidf = index[vec_tfidf]
    expected = [0.0, 0.256, 0.7022, 0.1524, 0.3334]
    for i, value in enumerate(expected):
        self.assertAlmostEqual(sims_tfidf[i], value, 2)