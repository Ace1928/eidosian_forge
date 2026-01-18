import logging
import unittest
import os
import bz2
import sys
import tempfile
import subprocess
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import word2vec, keyedvectors
from gensim.utils import check_output
from gensim.test.utils import (
def test_locking(self):
    """Test word2vec training doesn't change locked vectors."""
    corpus = LeeCorpus()
    for sg in range(2):
        model = word2vec.Word2Vec(vector_size=4, hs=1, negative=5, min_count=1, sg=sg, window=5)
        model.build_vocab(corpus)
        locked0 = np.copy(model.wv.vectors[0])
        unlocked1 = np.copy(model.wv.vectors[1])
        model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)
        model.wv.vectors_lockf[0] = 0.0
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        self.assertFalse((unlocked1 == model.wv.vectors[1]).all())
        self.assertTrue((locked0 == model.wv.vectors[0]).all())