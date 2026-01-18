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
def test_continuation_native(self):
    """Ensure that training has had a measurable effect."""
    native = load_native()
    self.model_structural_sanity(native)
    word = 'society'
    old_vector = native.wv.get_vector(word).tolist()
    native.train(list_corpus, total_examples=len(list_corpus), epochs=native.epochs)
    new_vector = native.wv.get_vector(word).tolist()
    self.assertNotEqual(old_vector, new_vector)
    self.model_structural_sanity(native)