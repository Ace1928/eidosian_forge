from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
def test_missing_string_doctag(self):
    """Test doc2vec doctag alternatives"""
    corpus = list(DocsLeeCorpus(True))
    corpus = corpus[0:10] + corpus
    model = doc2vec.Doc2Vec(min_count=1)
    model.build_vocab(corpus)
    self.assertRaises(KeyError, model.dv.__getitem__, 'not_a_tag')