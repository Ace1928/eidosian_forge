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
@unittest.skipIf(POT_EXT is False, 'POT not installed')
def test_identical_sentences(self):
    """Check that the distance from a sentence to itself is zero."""
    model = word2vec.Word2Vec(sentences, min_count=1)
    sentence = ['survey', 'user', 'computer', 'system', 'response', 'time']
    distance = model.wv.wmdistance(sentence, sentence)
    self.assertEqual(0.0, distance)