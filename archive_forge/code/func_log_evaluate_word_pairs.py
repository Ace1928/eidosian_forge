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
@staticmethod
def log_evaluate_word_pairs(pearson, spearman, oov, pairs):
    logger.info('Pearson correlation coefficient against %s: %.4f', pairs, pearson[0])
    logger.info('Spearman rank-order correlation coefficient against %s: %.4f', pairs, spearman[0])
    logger.info('Pairs with unknown words ratio: %.1f%%', oov)