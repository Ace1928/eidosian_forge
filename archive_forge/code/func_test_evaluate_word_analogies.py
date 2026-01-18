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
def test_evaluate_word_analogies(self):
    """Test that evaluating analogies on KeyedVectors give sane results"""
    model = word2vec.Word2Vec(LeeCorpus())
    score, sections = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
    score_cosmul, sections_cosmul = model.wv.evaluate_word_analogies(datapath('questions-words.txt'), similarity_function='most_similar_cosmul')
    self.assertEqual(score, score_cosmul)
    self.assertEqual(sections, sections_cosmul)
    self.assertGreaterEqual(score, 0.0)
    self.assertLessEqual(score, 1.0)
    self.assertGreater(len(sections), 0)
    first_section = sections[0]
    self.assertIn('section', first_section)
    self.assertIn('correct', first_section)
    self.assertIn('incorrect', first_section)