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
def test_cython_line_sentence_works_with_filename(self):
    """Does CythonLineSentence work with a filename argument?"""
    from gensim.models import word2vec_corpusfile
    with utils.open(datapath('lee_background.cor'), 'rb') as orig:
        sentences = word2vec_corpusfile.CythonLineSentence(datapath('lee_background.cor'))
        for words in sentences:
            self.assertEqual(words, orig.readline().split())