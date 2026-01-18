import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def index_to_dict(self):
    contiguous2id = {n: word_id for word_id, n in self.id2contiguous.items()}
    return {contiguous2id[n]: doc_id_set for n, doc_id_set in enumerate(self._inverted_index)}