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
def test_custom_tokenizer(self):
    """
        define a custom tokenizer function and use it
        """
    wc = self.corpus_class(self.enwiki, processes=1, tokenizer_func=custom_tokenizer, token_max_len=16, token_min_len=1, lower=False)
    row = wc.get_texts()
    list_tokens = next(row)
    self.assertTrue(u'Anarchism' in list_tokens)
    self.assertTrue(u'collectivization' in list_tokens)
    self.assertTrue(u'a' in list_tokens)
    self.assertTrue(u'i.e.' in list_tokens)