from .utils import _toposort, groupby
def supercedes(a, b):
    """ A is consistent and strictly more specific than B """
    return len(a) == len(b) and all(map(issubclass, a, b))