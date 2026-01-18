from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
def close_enough(x, y):
    """Verify that two sequences of n-gram association values are within
    _EPSILON of each other.
    """
    return all((abs(x1[1] - y1[1]) <= _EPSILON for x1, y1 in zip(x, y)))