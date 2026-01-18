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
def get_tfidf_matrix_t(cnts):
    """
    Convert the word count matrix into tfidf one (torch version).

    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = term frequency in document
    * N = number of documents
    * Nt = number of occurences of term in all documents
    """
    Nt = get_doc_freqs_t(cnts)
    idft = ((cnts.size(1) - Nt + 0.5) / (Nt + 0.5)).log()
    idft[idft < 0] = 0
    tft = sparse_log1p(cnts)
    inds, vals = (tft._indices(), tft._values())
    for i, ind in enumerate(inds[0]):
        vals[i] *= idft[ind]
    tfidft = tft
    return tfidft