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
def test_most_similar_cosmul(self):
    self.assertEqual(len(self.test_model.wv.most_similar_cosmul(positive=['the', 'and'], topn=5)), 5)
    self.assertEqual(self.test_model.wv.most_similar_cosmul('the'), self.test_model.wv.most_similar_cosmul(positive=['the']))
    self.assertEqual(len(self.test_model.wv.most_similar_cosmul(['night', 'nights'], topn=5)), 5)
    self.assertEqual(self.test_model.wv.most_similar_cosmul('nights'), self.test_model.wv.most_similar_cosmul(positive=['nights']))
    self.assertEqual(self.test_model.wv.most_similar_cosmul('the', 'and'), self.test_model.wv.most_similar_cosmul(positive=['the'], negative=['and']))