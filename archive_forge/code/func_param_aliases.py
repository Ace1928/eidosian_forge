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
def param_aliases(d):
    """
    Called from __setstate__ in LabelledData in order to load
    old pickles with outdated parameter names.

    Warning: We want to keep pickle hacking to a minimum!
    """
    for old, new in ALIASES.items():
        old_param = f'_{old}_param_value'
        new_param = f'_{new}_param_value'
        if old_param in d:
            d[new_param] = d.pop(old_param)
    return d