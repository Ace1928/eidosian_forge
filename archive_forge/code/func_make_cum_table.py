from the disk or network on-the-fly, without loading your entire corpus into RAM.
from __future__ import division  # py3 "true division"
import logging
import sys
import os
import heapq
from timeit import default_timer
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from types import GeneratorType
import threading
import itertools
import copy
from queue import Queue, Empty
from numpy import float32 as REAL
import numpy as np
from gensim.utils import keep_vocab_item, call_on_class_only, deprecated
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
from gensim import utils, matutils
from gensim.models.keyedvectors import Vocab  # noqa
from smart_open.compression import get_supported_extensions
def make_cum_table(self, domain=2 ** 31 - 1):
    """Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.

        To draw a word index, choose a random integer up to the maximum value in the table (cum_table[-1]),
        then finding that integer's sorted insertion point (as if by `bisect_left` or `ndarray.searchsorted()`).
        That insertion point is the drawn index, coming up in proportion equal to the increment at that slot.

        """
    vocab_size = len(self.wv.index_to_key)
    self.cum_table = np.zeros(vocab_size, dtype=np.uint32)
    train_words_pow = 0.0
    for word_index in range(vocab_size):
        count = self.wv.get_vecattr(word_index, 'count')
        train_words_pow += count ** float(self.ns_exponent)
    cumulative = 0.0
    for word_index in range(vocab_size):
        count = self.wv.get_vecattr(word_index, 'count')
        cumulative += count ** float(self.ns_exponent)
        self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
    if len(self.cum_table) > 0:
        assert self.cum_table[-1] == domain