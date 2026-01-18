import operator
import warnings
def custom_distance(file):
    data = {}
    with open(file) as infile:
        for l in infile:
            labelA, labelB, dist = l.strip().split('\t')
            labelA = frozenset([labelA])
            labelB = frozenset([labelB])
            data[frozenset([labelA, labelB])] = float(dist)
    return lambda x, y: data[frozenset([x, y])]