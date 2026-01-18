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
def test_line_sentence_works_with_compressed_file(self):
    """Does LineSentence work with a compressed file object argument?"""
    with utils.open(datapath('head500.noblanks.cor'), 'rb') as orig:
        sentences = word2vec.LineSentence(bz2.BZ2File(datapath('head500.noblanks.cor.bz2')))
        for words in sentences:
            self.assertEqual(words, utils.to_unicode(orig.readline()).split())