import os
import sys
import time
import subprocess
from tempfile import NamedTemporaryFile
import low_index
from low_index import permutation_reps
from low_index import benchmark_util
def run_example_low_level(ex):
    short_relators = low_index.spin_short([low_index.parse_word(rank=ex['rank'], word=relator) for relator in ex['short relators']], max_degree=ex['index'])
    long_relators = [low_index.parse_word(rank=ex['rank'], word=relator) for relator in ex['long relators']]
    if low_level_multi_threaded:
        tree = low_index.SimsTreeMultiThreaded(rank=ex['rank'], max_degree=ex['index'], short_relators=short_relators, long_relators=long_relators, num_threads=low_index.hardware_concurrency())
    else:
        tree = low_index.SimsTree(rank=ex['rank'], max_degree=ex['index'], short_relators=short_relators, long_relators=long_relators)
    return len(tree.list())