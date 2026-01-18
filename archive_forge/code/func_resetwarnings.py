import sys
def resetwarnings():
    """Clear the list of warning filters, so that no filters are active."""
    filters[:] = []
    _filters_mutated()