from collections import namedtuple
import unittest
import logging
import numpy as np
import pytest
from scipy.spatial.distance import cosine
from gensim.models.doc2vec import Doc2Vec
from gensim import utils
from gensim.models import translation_matrix
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
def read_sentiment_docs(filename):
    sentiment_document = namedtuple('SentimentDocument', 'words tags')
    alldocs = []
    with utils.open(filename, mode='rb', encoding='utf-8') as alldata:
        for line_no, line in enumerate(alldata):
            tokens = utils.to_unicode(line).split()
            words = tokens
            tags = str(line_no)
            alldocs.append(sentiment_document(words, tags))
    return alldocs