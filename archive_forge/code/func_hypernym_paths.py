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
def hypernym_paths(self):
    """
        Get the path(s) from this synset to the root, where each path is a
        list of the synset nodes traversed on the way to the root.

        :return: A list of lists, where each list gives the node sequence
           connecting the initial ``Synset`` node and a root node.
        """
    paths = []
    hypernyms = self.hypernyms() + self.instance_hypernyms()
    if len(hypernyms) == 0:
        paths = [[self]]
    for hypernym in hypernyms:
        for ancestor_list in hypernym.hypernym_paths():
            ancestor_list.append(self)
            paths.append(ancestor_list)
    return paths