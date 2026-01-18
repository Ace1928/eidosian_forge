import sys
from . import context
def ensure_multiprocessing():
    from ._ext import ensure_multiprocessing
    return ensure_multiprocessing()