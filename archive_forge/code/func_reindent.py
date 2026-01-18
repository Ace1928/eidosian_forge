import gc
import itertools
import sys
import time
def reindent(src, indent):
    """Helper to reindent a multi-line statement."""
    return src.replace('\n', '\n' + ' ' * indent)