import builtins
import datetime as dt
import re
import weakref
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import partial
from itertools import chain
from operator import itemgetter
import numpy as np
import param
from . import util
from .accessors import Apply, Opts, Redim
from .options import Options, Store, cleanup_custom_options
from .pprint import PrettyPrinter
from .tree import AttrTree
from .util import bytes_to_unicode
def process_dimensions(kdims, vdims):
    """Converts kdims and vdims to Dimension objects.

    Args:
        kdims: List or single key dimension(s) specified as strings,
            tuples dicts or Dimension objects.
        vdims: List or single value dimension(s) specified as strings,
            tuples dicts or Dimension objects.

    Returns:
        Dictionary containing kdims and vdims converted to Dimension
        objects:

        {'kdims': [Dimension('x')], 'vdims': [Dimension('y')]
    """
    dimensions = {}
    for group, dims in [('kdims', kdims), ('vdims', vdims)]:
        if dims is None:
            continue
        elif isinstance(dims, (tuple, str, Dimension, dict)):
            dims = [dims]
        elif not isinstance(dims, list):
            raise ValueError('{} argument expects a Dimension or list of dimensions, specified as tuples, strings, dictionaries or Dimension instances, not a {} type. Ensure you passed the data as the first argument.'.format(group, type(dims).__name__))
        dimensions[group] = [asdim(d) for d in dims]
    return dimensions