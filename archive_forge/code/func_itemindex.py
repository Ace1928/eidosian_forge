import warnings
from Bio import BiopythonDeprecationWarning
def itemindex(values):
    """Return a dictionary of values with their sequence offset as keys."""
    d = {}
    entries = enumerate(values[::-1])
    n = len(values) - 1
    for index, key in entries:
        d[key] = n - index
    return d