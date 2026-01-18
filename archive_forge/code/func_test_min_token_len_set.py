from __future__ import unicode_literals
import codecs
import itertools
import logging
import os
import os.path
import tempfile
import unittest
import numpy as np
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
from gensim.interfaces import TransformedCorpus
from gensim.utils import to_unicode
from gensim.test.utils import datapath, get_tmpfile, common_corpus
def test_min_token_len_set(self):
    """
        Set the parameter token_min_len to 1 and check that 'a' as a token exists
        """
    corpus = self.corpus_class(self.enwiki, processes=1, token_min_len=1)
    self.assertTrue(u'a' in next(corpus.get_texts()))