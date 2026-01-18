import logging
import unittest
import math
import os
import numpy
import scipy
from gensim import utils
from gensim.corpora import Dictionary
from gensim.models import word2vec
from gensim.models import doc2vec
from gensim.models import KeyedVectors
from gensim.models import TfidfModel
from gensim import matutils, similarities
from gensim.models import Word2Vec, FastText
from gensim.test.utils import (
from gensim.similarities import UniformTermSimilarityIndex
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import LevenshteinSimilarityIndex
from gensim.similarities.docsim import _nlargest
from gensim.similarities.fastss import editdist
def assertLoadedIndexEqual(self, index, model):
    from gensim.similarities.nmslib import NmslibIndexer
    fname = get_tmpfile('gensim_similarities.tst.pkl')
    index.save(fname)
    index2 = NmslibIndexer.load(fname)
    index2.model = model
    self.assertEqual(index.labels, index2.labels)
    self.assertEqual(index.index_params, index2.index_params)
    self.assertEqual(index.query_time_params, index2.query_time_params)