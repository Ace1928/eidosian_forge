import torch
import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter
from . import utils
from .doc_db import DocDB
from . import tokenizers
import parlai.utils.logging as logging
def count_text(ngram, hash_size, doc_id, text=None):
    """
    Compute hashed ngram counts of text.
    """
    row, col, data = ([], [], [])
    tokens = tokenize(utils.normalize(text))
    ngrams = tokens.ngrams(n=ngram, uncased=True, filter_fn=utils.filter_ngram)
    counts = Counter([utils.hash(gram, hash_size) for gram in ngrams])
    row.extend(counts.keys())
    col.extend([doc_id] * len(counts))
    data.extend(counts.values())
    return (row, col, data)