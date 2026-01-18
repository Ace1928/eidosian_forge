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
def create_binary_tree(self):
    """Create a `binary Huffman tree <https://en.wikipedia.org/wiki/Huffman_coding>`_ using stored vocabulary
        word counts. Frequent words will have shorter binary codes.
        Called internally from :meth:`~gensim.models.word2vec.Word2VecVocab.build_vocab`.

        """
    _assign_binary_codes(self.wv)