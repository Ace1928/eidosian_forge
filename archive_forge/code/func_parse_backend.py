from collections import namedtuple
from decimal import Decimal
import numpy as np
from . import backends, blas, helpers, parser, paths, sharing
def parse_backend(arrays, backend):
    """Find out what backend we should use, dipatching based on the first
    array if ``backend='auto'`` is specified.
    """
    if backend != 'auto':
        return backend
    backend = infer_backend(arrays[0])
    if not backends.has_tensordot(backend):
        return 'numpy'
    return backend