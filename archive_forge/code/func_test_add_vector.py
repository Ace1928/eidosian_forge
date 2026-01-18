from __future__ import division
import gzip
import io
import logging
import unittest
import os
import shutil
import subprocess
import struct
import sys
import numpy as np
import pytest
from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim, FastTextKeyedVectors, _unpack
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import (
from gensim.test.test_word2vec import TestWord2VecModel
import gensim.models._fasttext_bin
from gensim.models.fasttext_inner import compute_ngrams, compute_ngrams_bytes, ft_hash_bytes
import gensim.models.fasttext
def test_add_vector(self):
    wv = FastTextKeyedVectors(vector_size=2, min_n=3, max_n=6, bucket=2000000)
    wv.add_vector('test_key', np.array([0, 0]))
    self.assertEqual(wv.key_to_index['test_key'], 0)
    self.assertEqual(wv.index_to_key[0], 'test_key')
    self.assertTrue(np.all(wv.vectors[0] == np.array([0, 0])))