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
def test_get_vocab_word_vecs(self):
    model = FT_gensim(vector_size=12, min_count=1, seed=42, bucket=BUCKET)
    model.build_vocab(sentences)
    original_syn0_vocab = np.copy(model.wv.vectors_vocab)
    model.wv.adjust_vectors()
    self.assertTrue(np.all(np.equal(model.wv.vectors_vocab, original_syn0_vocab)))