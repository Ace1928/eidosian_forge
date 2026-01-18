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
def assertAllSimilaritiesDisableIndexer(self, model, wv, index):
    vector = wv.get_normed_vectors()[0]
    approx_similarities = model.most_similar([vector], topn=None, indexer=index)
    exact_similarities = model.most_similar(positive=[vector], topn=None)
    self.assertEqual(approx_similarities, exact_similarities)
    self.assertEqual(len(approx_similarities), len(wv.vectors))