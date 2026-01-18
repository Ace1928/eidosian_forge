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
def test_doc_freq_and_token2id_for_several_docs_with_one_word(self):
    texts = [['human'], ['human']]
    d = Dictionary(texts)
    expected = {0: 2}
    self.assertEqual(d.dfs, expected)
    expected = {'human': 0}
    self.assertEqual(d.token2id, expected)
    texts = [['human'], ['human'], ['human']]
    d = Dictionary(texts)
    expected = {0: 3}
    self.assertEqual(d.dfs, expected)
    expected = {'human': 0}
    self.assertEqual(d.token2id, expected)
    texts = [['human'], ['human'], ['human'], ['human']]
    d = Dictionary(texts)
    expected = {0: 4}
    self.assertEqual(d.dfs, expected)
    expected = {'human': 0}
    self.assertEqual(d.token2id, expected)