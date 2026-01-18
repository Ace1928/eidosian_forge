import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def reply_to_master(self):
    logger.info('serializing accumulator to return to master...')
    self.output_q.put(self.accumulator, block=False)
    logger.info('accumulator serialized')