import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
def check_moved(self):
    """Update shard locations, for case where the server prefix location changed on the filesystem."""
    dirname = os.path.dirname(self.output_prefix)
    for shard in self.shards:
        shard.dirname = dirname