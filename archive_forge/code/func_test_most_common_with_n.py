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
def test_most_common_with_n(self):
    texts = [['human', 'human', 'human', 'computer', 'computer', 'interface', 'interface']]
    d = Dictionary(texts)
    expected = [('human', 3), ('computer', 2)]
    assert d.most_common(n=2) == expected