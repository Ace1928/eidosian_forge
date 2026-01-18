from collections import defaultdict
import math
import unittest
from gensim.models.bm25model import BM25ABC
from gensim.models import OkapiBM25Model, LuceneBM25Model, AtireBM25Model
from gensim.corpora import Dictionary
def get_term_weights(self, num_tokens, term_frequencies, idfs):
    return term_frequencies