from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def logsumexp2(arr):
    max_ = arr.max()
    return np.log2(np.sum(2 ** (arr - max_))) + max_