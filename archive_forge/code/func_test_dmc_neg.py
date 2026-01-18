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
def test_dmc_neg(self):
    """Test DM/concatenate doc2vec training."""
    model = doc2vec.Doc2Vec(list_corpus, dm=1, dm_concat=1, vector_size=24, window=4, hs=0, negative=10, alpha=0.05, min_count=2, epochs=20)
    self.model_sanity(model)