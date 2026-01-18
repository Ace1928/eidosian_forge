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
def test_filter_most_frequent(self):
    d = Dictionary(self.texts)
    d.filter_n_most_frequent(4)
    expected = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2}
    self.assertEqual(d.dfs, expected)