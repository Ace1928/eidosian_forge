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
def pprint_value_string(self, value):
    """Pretty print the dimension value and unit with title_format

        Args:
            value: Dimension value to format

        Returns:
            Formatted dimension value string with unit
        """
    unit = '' if self.unit is None else ' ' + bytes_to_unicode(self.unit)
    value = self.pprint_value(value)
    return title_format.format(name=bytes_to_unicode(self.label), val=value, unit=unit)