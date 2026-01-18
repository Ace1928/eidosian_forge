import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
def nbow(document):
    d = zeros(vocab_len, dtype=double)
    nbow = dictionary.doc2bow(document)
    doc_len = len(document)
    for idx, freq in nbow:
        d[idx] = freq / float(doc_len)
    return d