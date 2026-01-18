import os
import sys
import time
import subprocess
from tempfile import NamedTemporaryFile
import low_index
from low_index import permutation_reps
from low_index import benchmark_util
def run_regina(ex):
    print('%s; index = %d.' % (ex['group'], ex['index']))
    if ex['index'] > 7:
        print('Skipping because regina requires index <= 7')
        sys.stdout.flush()
        return
    start = time.time()
    G = regina.GroupPresentation(ex['rank'], ex['short relators'] + ex['long relators'])
    n = 1
    for d in range(2, ex['index'] + 1):
        n += len(G.enumerateCovers(d))
    elapsed = time.time() - start
    print('%d subgroups' % n)
    print('%.3fs' % elapsed)
    sys.stdout.flush()