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
def predict_output_word(self, context_words_list, topn=10):
    """Get the probability distribution of the center word given context words.

        Note this performs a CBOW-style propagation, even in SG models,
        and doesn't quite weight the surrounding words the same as in
        training -- so it's just one crude way of using a trained model
        as a predictor.

        Parameters
        ----------
        context_words_list : list of (str and/or int)
            List of context words, which may be words themselves (str)
            or their index in `self.wv.vectors` (int).
        topn : int, optional
            Return `topn` words and their probabilities.

        Returns
        -------
        list of (str, float)
            `topn` length list of tuples of (word, probability).

        """
    if not self.negative:
        raise RuntimeError('We have currently only implemented predict_output_word for the negative sampling scheme, so you need to have run word2vec with negative > 0 for this to work.')
    if not hasattr(self.wv, 'vectors') or not hasattr(self, 'syn1neg'):
        raise RuntimeError('Parameters required for predicting the output words not found.')
    word2_indices = [self.wv.get_index(w) for w in context_words_list if w in self.wv]
    if not word2_indices:
        logger.warning('All the input context words are out-of-vocabulary for the current model.')
        return None
    l1 = np.sum(self.wv.vectors[word2_indices], axis=0)
    if word2_indices and self.cbow_mean:
        l1 /= len(word2_indices)
    prob_values = np.exp(np.dot(l1, self.syn1neg.T))
    prob_values /= np.sum(prob_values)
    top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
    return [(self.wv.index_to_key[index1], prob_values[index1]) for index1 in top_indices]