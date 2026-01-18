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
def all_eng_synsets(self, pos=None):
    if pos is None:
        pos_tags = self._FILEMAP.keys()
    else:
        pos_tags = [pos]
    cache = self._synset_offset_cache
    from_pos_and_line = self._synset_from_pos_and_line
    for pos_tag in pos_tags:
        if pos_tag == ADJ_SAT:
            pos_file = ADJ
        else:
            pos_file = pos_tag
        fileid = 'data.%s' % self._FILEMAP[pos_file]
        data_file = self.open(fileid)
        try:
            offset = data_file.tell()
            line = data_file.readline()
            while line:
                if not line[0].isspace():
                    if offset in cache[pos_tag]:
                        synset = cache[pos_tag][offset]
                    else:
                        synset = from_pos_and_line(pos_tag, line)
                        cache[pos_tag][offset] = synset
                    if pos_tag == ADJ_SAT and synset._pos == ADJ_SAT:
                        yield synset
                    elif pos_tag != ADJ_SAT:
                        yield synset
                offset = data_file.tell()
                line = data_file.readline()
        except:
            data_file.close()
            raise
        else:
            data_file.close()