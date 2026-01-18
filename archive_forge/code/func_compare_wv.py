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
def compare_wv(a, b, t):
    a_count = {key: a.get_vecattr(key, 'count') for key in a.key_to_index}
    b_count = {key: b.get_vecattr(key, 'count') for key in b.key_to_index}
    t.assertEqual(a_count, b_count)
    t.assertEqual(a.vectors.shape, b.vectors.shape)
    t.assertEqual(a.vectors_vocab.shape, b.vectors_vocab.shape)