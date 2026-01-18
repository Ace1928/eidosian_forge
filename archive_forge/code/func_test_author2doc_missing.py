import logging
import unittest
import numbers
from os import remove
import numpy as np
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import atmodel
from gensim import matutils
from gensim.test import basetmtests
from gensim.test.utils import (datapath,
from gensim.matutils import jensen_shannon
def test_author2doc_missing(self):
    model = self.class_(corpus, author2doc=author2doc, doc2author=doc2author, id2word=dictionary, num_topics=2, random_state=0)
    model2 = self.class_(corpus, doc2author=doc2author, id2word=dictionary, num_topics=2, random_state=0)
    jill_topics = model.get_author_topics('jill')
    jill_topics2 = model2.get_author_topics('jill')
    jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
    jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)
    self.assertTrue(np.allclose(jill_topics, jill_topics2))