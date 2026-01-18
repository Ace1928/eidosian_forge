import itertools as it
from abc import ABCMeta, abstractmethod
from nltk.tbl.feature import Feature
from nltk.tbl.rule import Rule
def get_neighborhood(self, tokens, index):
    neighborhood = {index}
    allpositions = [0] + [p for feat in self._features for p in feat.positions]
    start, end = (min(allpositions), max(allpositions))
    s = max(0, index + -end)
    e = min(index + -start + 1, len(tokens))
    for i in range(s, e):
        neighborhood.add(i)
    return neighborhood