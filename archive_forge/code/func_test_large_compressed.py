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
def test_large_compressed(self):
    if self.cls == similarities.WmdSimilarity and (not POT_EXT):
        self.skipTest('POT not installed')
    fname = get_tmpfile('gensim_similarities.tst.pkl.gz')
    index = self.factoryMethod()
    index.save(fname, sep_limit=0)
    index2 = self.cls.load(fname, mmap=None)
    if self.cls == similarities.Similarity:
        self.assertTrue(len(index.shards) == len(index2.shards))
        index.destroy()
    else:
        if isinstance(index, similarities.SparseMatrixSimilarity):
            index.index = index.index.todense()
            index2.index = index2.index.todense()
        self.assertTrue(numpy.allclose(index.index, index2.index))
        self.assertEqual(index.num_best, index2.num_best)