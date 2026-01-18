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
def test_fb(self):
    """Test against results from Facebook's implementation."""
    with utils.open(datapath('fb-ngrams.txt'), 'r', encoding='utf-8') as fin:
        fb = dict(_read_fb(fin))
    for word, expected in fb.items():
        actual = compute_ngrams(word, 3, 6)
        self.assertEqual(sorted(expected), sorted(actual))