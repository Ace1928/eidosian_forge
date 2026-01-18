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
def get_dimension(self, dimension, default=None, strict=False):
    """Get a Dimension object by name or index.

        Args:
            dimension: Dimension to look up by name or integer index
            default (optional): Value returned if Dimension not found
            strict (bool, optional): Raise a KeyError if not found

        Returns:
            Dimension object for the requested dimension or default
        """
    if dimension is not None and (not isinstance(dimension, (int, str, Dimension))):
        raise TypeError('Dimension lookup supports int, string, and Dimension instances, cannot lookup Dimensions using %s type.' % type(dimension).__name__)
    all_dims = self.dimensions()
    if isinstance(dimension, int):
        if 0 <= dimension < len(all_dims):
            return all_dims[dimension]
        elif strict:
            raise KeyError(f'Dimension {dimension!r} not found')
        else:
            return default
    if isinstance(dimension, Dimension):
        dims = [d for d in all_dims if dimension == d]
        if strict and (not dims):
            raise KeyError(f'{dimension!r} not found.')
        elif dims:
            return dims[0]
        else:
            return None
    else:
        dimension = dimension_name(dimension)
        name_map = {dim.spec: dim for dim in all_dims}
        name_map.update({dim.name: dim for dim in all_dims})
        name_map.update({dim.label: dim for dim in all_dims})
        name_map.update({util.dimension_sanitizer(dim.name): dim for dim in all_dims})
        if strict and dimension not in name_map:
            raise KeyError(f'Dimension {dimension!r} not found.')
        else:
            return name_map.get(dimension, default)