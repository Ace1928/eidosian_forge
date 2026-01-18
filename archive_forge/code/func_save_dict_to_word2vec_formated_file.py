import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def save_dict_to_word2vec_formated_file(fname, word2vec_dict):
    with gensim.utils.open(fname, 'wb') as f:
        num_words = len(word2vec_dict)
        vector_length = len(list(word2vec_dict.values())[0])
        header = '%d %d\n' % (num_words, vector_length)
        f.write(header.encode(encoding='ascii'))
        for word, vector in word2vec_dict.items():
            f.write(word.encode())
            f.write(' '.encode())
            f.write(np.array(vector).astype(np.float32).tobytes())