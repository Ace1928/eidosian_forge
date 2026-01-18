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
def test_max_final_vocab(self):
    model = word2vec.Word2Vec(vector_size=10, max_final_vocab=4, min_count=4, sample=0)
    model.scan_vocab(sentences)
    reported_values = model.prepare_vocab()
    self.assertEqual(reported_values['drop_unique'], 11)
    self.assertEqual(reported_values['retain_total'], 4)
    self.assertEqual(reported_values['num_retained_words'], 1)
    self.assertEqual(model.effective_min_count, 4)
    model = word2vec.Word2Vec(vector_size=10, max_final_vocab=4, min_count=2, sample=0)
    model.scan_vocab(sentences)
    reported_values = model.prepare_vocab()
    self.assertEqual(reported_values['drop_unique'], 8)
    self.assertEqual(reported_values['retain_total'], 13)
    self.assertEqual(reported_values['num_retained_words'], 4)
    self.assertEqual(model.effective_min_count, 3)