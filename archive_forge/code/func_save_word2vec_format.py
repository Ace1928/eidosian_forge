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
def save_word2vec_format(self, fname, doctag_vec=False, word_vec=True, prefix='*dt_', fvocab=None, binary=False):
    """Store the input-hidden weight matrix in the same format used by the original C word2vec-tool.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        doctag_vec : bool, optional
            Indicates whether to store document vectors.
        word_vec : bool, optional
            Indicates whether to store word vectors.
        prefix : str, optional
            Uniquely identifies doctags from word vocab, and avoids collision in case of repeated string in doctag
            and word vocab.
        fvocab : str, optional
            Optional file path used to save the vocabulary.
        binary : bool, optional
            If True, the data will be saved in binary word2vec format, otherwise - will be saved in plain text.

        """
    total_vec = None
    if word_vec:
        if doctag_vec:
            total_vec = len(self.wv) + len(self.dv)
        self.wv.save_word2vec_format(fname, fvocab, binary, total_vec)
    if doctag_vec:
        write_header = True
        append = False
        if word_vec:
            write_header = False
            append = True
        self.dv.save_word2vec_format(fname, prefix=prefix, fvocab=fvocab, binary=binary, write_header=write_header, append=append, sort_attr='doc_count')