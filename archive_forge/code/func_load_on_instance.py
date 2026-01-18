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
def load_on_instance():
    tmpf = get_tmpfile('gensim_doc2vec.tst')
    model = doc2vec.Doc2Vec(DocsLeeCorpus(), min_count=1)
    model.save(tmpf)
    model = doc2vec.Doc2Vec()
    return model.load(tmpf)