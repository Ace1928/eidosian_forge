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
def test_get_author_topics(self):
    model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2, passes=100, random_state=np.random.seed(0))
    author_topics = []
    for a in model.id2author.values():
        author_topics.append(model.get_author_topics(a))
    for topic in author_topics:
        self.assertTrue(isinstance(topic, list))
        for k, v in topic:
            self.assertTrue(isinstance(k, int))
            self.assertTrue(isinstance(v, float))