from builtins import abs as _abs
def indexOf(a, b):
    """Return the first index of b in a."""
    for i, j in enumerate(a):
        if j is b or j == b:
            return i
    else:
        raise ValueError('sequence.index(x): x not in sequence')