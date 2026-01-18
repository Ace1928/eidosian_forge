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
def test_load_with_metadata(self):
    corpus = self.corpus_class(self.fname, article_min_tokens=0)
    corpus.metadata = True
    self.assertEqual(len(corpus), 9)
    docs = list(corpus)
    self.assertEqual(len(docs), 9)
    for i, docmeta in enumerate(docs):
        doc, metadata = docmeta
        article_no = i + 1
        self.assertEqual(metadata[0], str(article_no))
        self.assertEqual(metadata[1], 'Article%d' % article_no)