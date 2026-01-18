from __future__ import print_function
import logging
import os
import math
import time
import numpy
import scipy.sparse as sparse
import gensim
from gensim.corpora import IndexedCorpus
from gensim.interfaces import TransformedCorpus
def in_current(self, offset):
    """
        Determine whether the given offset falls within the current shard.

        """
    return self.current_offset <= offset and offset < self.offsets[self.current_shard_n + 1]