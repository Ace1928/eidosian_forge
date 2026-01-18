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
def test_bucket_ngrams(self):
    model = FT_gensim(vector_size=12, min_count=1, bucket=20)
    model.build_vocab(sentences)
    self.assertEqual(model.wv.vectors_ngrams.shape, (20, 12))
    model.build_vocab(new_sentences, update=True)
    self.assertEqual(model.wv.vectors_ngrams.shape, (20, 12))