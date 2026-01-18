from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def cpd(array, conditions, samples):
    d = {}
    for values, condition in zip(array, conditions):
        d[condition] = pd(values, samples)
    return DictionaryConditionalProbDist(d)