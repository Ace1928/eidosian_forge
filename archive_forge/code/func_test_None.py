import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def test_None(self):
    result = utils.is_corpus(None)
    expected = (False, None)
    self.assertEqual(expected, result)