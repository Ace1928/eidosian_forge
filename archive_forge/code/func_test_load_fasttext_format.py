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
def test_load_fasttext_format(self):
    try:
        model = gensim.models.fasttext.load_facebook_model(self.test_model_file)
    except Exception as exc:
        self.fail('Unable to load FastText model from file %s: %s' % (self.test_model_file, exc))
    vocab_size, model_size = (1762, 10)
    self.assertEqual(model.wv.vectors.shape, (vocab_size, model_size))
    self.assertEqual(len(model.wv), vocab_size, model_size)
    self.assertEqual(model.wv.vectors_ngrams.shape, (model.wv.bucket, model_size))
    expected_vec = [-0.57144, -0.0085561, 0.15748, -0.67855, -0.25459, -0.58077, -0.09913, 1.1447, 0.23418, 0.060007]
    actual_vec = model.wv['hundred']
    self.assertTrue(np.allclose(actual_vec, expected_vec, atol=0.0001))
    expected_vec_oov = [-0.21929, -0.53778, -0.22463, -0.41735, 0.71737, -1.59758, -0.24833, 0.62028, 0.53203, 0.77568]
    actual_vec_oov = model.wv['rejection']
    self.assertTrue(np.allclose(actual_vec_oov, expected_vec_oov, atol=0.0001))
    self.assertEqual(model.min_count, 5)
    self.assertEqual(model.window, 5)
    self.assertEqual(model.epochs, 5)
    self.assertEqual(model.negative, 5)
    self.assertEqual(model.sample, 0.0001)
    self.assertEqual(model.wv.bucket, 1000)
    self.assertEqual(model.wv.max_n, 6)
    self.assertEqual(model.wv.min_n, 3)
    self.assertEqual(model.wv.vectors.shape, (len(model.wv), model.vector_size))
    self.assertEqual(model.wv.vectors_ngrams.shape, (model.wv.bucket, model.vector_size))