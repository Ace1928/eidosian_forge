import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def not_in_vocab(self, words):
    uniq_words = set(utils.flatten(words))
    return set((word for word in uniq_words if word not in self.model))