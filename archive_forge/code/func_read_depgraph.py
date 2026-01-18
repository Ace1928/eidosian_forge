from itertools import chain
from nltk.internals import Counter
@staticmethod
def read_depgraph(depgraph):
    return FStructure._read_depgraph(depgraph.root, depgraph)