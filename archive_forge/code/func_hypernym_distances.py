import math
import os
import re
import warnings
from collections import defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from operator import itemgetter
from nltk.corpus.reader import CorpusReader
from nltk.internals import deprecated
from nltk.probability import FreqDist
from nltk.util import binary_search_file as _binary_search_file
def hypernym_distances(self, distance=0, simulate_root=False):
    """
        Get the path(s) from this synset to the root, counting the distance
        of each node from the initial node on the way. A set of
        (synset, distance) tuples is returned.

        :type distance: int
        :param distance: the distance (number of edges) from this hypernym to
            the original hypernym ``Synset`` on which this method was called.
        :return: A set of ``(Synset, int)`` tuples where each ``Synset`` is
           a hypernym of the first ``Synset``.
        """
    distances = {(self, distance)}
    for hypernym in self._hypernyms() + self._instance_hypernyms():
        distances |= hypernym.hypernym_distances(distance + 1, simulate_root=False)
    if simulate_root:
        fake_synset = Synset(None)
        fake_synset._name = '*ROOT*'
        fake_synset_distance = max(distances, key=itemgetter(1))[1]
        distances.add((fake_synset, fake_synset_distance + 1))
    return distances