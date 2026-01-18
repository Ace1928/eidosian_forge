import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def parse_num_atoms(s):
    num_atoms = int(s)
    if num_atoms < 2:
        raise argparse.ArgumentTypeError('must be at least 2, not %s' % s)
    return num_atoms