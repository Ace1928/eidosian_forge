import warnings
import numpy
def sortByDimension(self, nodes, i):
    """Sorts the list of nodes by the i-th value of the contained points."""
    decorated = [(node.cargo[i], node) for node in nodes]
    decorated.sort()
    nodes[:] = [node for _, node in decorated]