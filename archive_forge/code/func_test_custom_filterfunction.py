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
def test_custom_filterfunction(self):

    def reject_all(elem, *args, **kwargs):
        return False
    corpus = self.corpus_class(self.enwiki, filter_articles=reject_all)
    texts = corpus.get_texts()
    self.assertFalse(any(texts))

    def keep_some(elem, title, *args, **kwargs):
        return title[0] == 'C'
    corpus = self.corpus_class(self.enwiki, filter_articles=reject_all)
    corpus.metadata = True
    texts = corpus.get_texts()
    for text, (pageid, title) in texts:
        self.assertEquals(title[0], 'C')