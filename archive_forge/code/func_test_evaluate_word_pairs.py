import logging
import unittest
import os
import bz2
import sys
import tempfile
import subprocess
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import word2vec, keyedvectors
from gensim.utils import check_output
from gensim.test.utils import (
def test_evaluate_word_pairs(self):
    """Test Spearman and Pearson correlation coefficients give sane results on similarity datasets"""
    corpus = word2vec.LineSentence(datapath('head500.noblanks.cor.bz2'))
    model = word2vec.Word2Vec(corpus, min_count=3, epochs=20)
    correlation = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
    pearson = correlation[0][0]
    spearman = correlation[1][0]
    oov = correlation[2]
    self.assertTrue(0.1 < pearson < 1.0, f'pearson {pearson} not between 0.1 & 1.0')
    self.assertTrue(0.1 < spearman < 1.0, f'spearman {spearman} not between 0.1 and 1.0')
    self.assertTrue(0.0 <= oov < 90.0, f'OOV {oov} not between 0.0 and 90.0')