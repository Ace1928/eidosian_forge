from collections.abc import Mapping
from itertools import chain
import logging
import unittest
import codecs
import os
import os.path
import scipy
import gensim
from gensim.corpora import Dictionary
from gensim.utils import to_utf8
from gensim.test.utils import get_tmpfile, common_texts
def test_doc_freq_one_doc(self):
    texts = [['human', 'interface', 'computer']]
    d = Dictionary(texts)
    expected = {0: 1, 1: 1, 2: 1}
    self.assertEqual(d.dfs, expected)