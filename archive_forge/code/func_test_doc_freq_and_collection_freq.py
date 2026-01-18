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
def test_doc_freq_and_collection_freq(self):
    texts = [['human', 'human', 'human']]
    d = Dictionary(texts)
    self.assertEqual(d.cfs, {0: 3})
    self.assertEqual(d.dfs, {0: 1})
    texts = [['human', 'human'], ['human']]
    d = Dictionary(texts)
    self.assertEqual(d.cfs, {0: 3})
    self.assertEqual(d.dfs, {0: 2})
    texts = [['human'], ['human'], ['human']]
    d = Dictionary(texts)
    self.assertEqual(d.cfs, {0: 3})
    self.assertEqual(d.dfs, {0: 3})