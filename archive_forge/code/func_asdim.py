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
def asdim(dimension):
    """Convert the input to a Dimension.

    Args:
        dimension: tuple, dict or string type to convert to Dimension

    Returns:
        A Dimension object constructed from the dimension spec. No
        copy is performed if the input is already a Dimension.
    """
    return dimension if isinstance(dimension, Dimension) else Dimension(dimension)