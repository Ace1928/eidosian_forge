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
def map_to_one(self):
    synset_to_many = self.map_to_many()
    synset_to_one = {}
    for source in synset_to_many:
        candidates_bag = synset_to_many[source]
        if candidates_bag:
            candidates_set = set(candidates_bag)
            if len(candidates_set) == 1:
                target = candidates_bag[0]
            else:
                counts = []
                for candidate in candidates_set:
                    counts.append((candidates_bag.count(candidate), candidate))
                self.splits[source] = counts
                target = max(counts)[1]
            synset_to_one[source] = target
            if source[-1] == 's':
                synset_to_one[f'{source[:-1]}a'] = target
        else:
            self.nomap.append(source)
    return synset_to_one