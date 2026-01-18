import logging
import os
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from timeit import default_timer
from dataclasses import dataclass
from numpy import zeros, float32 as REAL, vstack, integer, dtype
import numpy as np
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.utils import deprecated
from gensim.models import Word2Vec, FAST_VERSION  # noqa: F401
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
def d2v_train_epoch_dm(model, corpus_file, offset, start_doctag, _cython_vocab, _cur_epoch, _expected_examples, _expected_words, work, _neu1, docvecs_count, word_vectors=None, word_locks=None, learn_doctags=True, learn_words=True, learn_hidden=True, doctag_vectors=None, doctag_locks=None):
    raise NotImplementedError('Training with corpus_file argument is not supported.')