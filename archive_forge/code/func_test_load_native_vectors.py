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
def test_load_native_vectors(self):
    cap_path = datapath('crime-and-punishment.bin')
    fbkv = gensim.models.fasttext.load_facebook_vectors(cap_path)
    self.assertFalse('landlord' in fbkv.key_to_index)
    self.assertTrue('landlady' in fbkv.key_to_index)
    oov_vector = fbkv['landlord']
    iv_vector = fbkv['landlady']
    self.assertFalse(np.allclose(oov_vector, iv_vector))