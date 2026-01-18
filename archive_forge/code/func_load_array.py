import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def load_array(self, array):
    """Load an sequence of annotation results, appending to any data already loaded.

        The argument is a sequence of 3-tuples, each representing a coder's labeling of an item:
            (coder,item,label)
        """
    for coder, item, labels in array:
        self.C.add(coder)
        self.K.add(labels)
        self.I.add(item)
        self.data.append({'coder': coder, 'labels': labels, 'item': item})