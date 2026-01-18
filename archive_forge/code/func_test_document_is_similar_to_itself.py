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
def test_document_is_similar_to_itself(self):
    approx_neighbors = self.index.most_similar(self.vector, 1)
    doc, similarity = approx_neighbors[0]
    self.assertEqual(doc, 0)
    self.assertAlmostEqual(similarity, 1.0, places=2)