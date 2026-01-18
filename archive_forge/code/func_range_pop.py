from contextlib import contextmanager
def range_pop():
    """Pop a range off of a stack of nested range spans.  Returns the  zero-based depth of the range that is ended."""
    return _nvtx.rangePop()