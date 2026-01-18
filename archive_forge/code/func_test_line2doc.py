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
def test_line2doc(self):
    super(TestMalletCorpus, self).test_line2doc()
    fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
    id2word = {1: 'mom', 2: 'window'}
    corpus = self.corpus_class(fname, id2word=id2word, metadata=True)
    corpus.use_wordids = False
    doc, (docid, doclang) = corpus.line2doc(self.CORPUS_LINE)
    self.assertEqual(docid, '#3')
    self.assertEqual(doclang, 'lang')
    self.assertEqual(sorted(doc), [('mom', 1), ('was', 1), ('wash', 1), ('washed', 1), ('window', 2)])
    corpus.use_wordids = True
    doc, (docid, doclang) = corpus.line2doc(self.CORPUS_LINE)
    self.assertEqual(docid, '#3')
    self.assertEqual(doclang, 'lang')
    self.assertEqual(sorted(doc), [(1, 1), (2, 2)])