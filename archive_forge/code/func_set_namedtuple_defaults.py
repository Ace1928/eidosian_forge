from collections import deque, OrderedDict
from typing import Union, Optional, Set, Any, Dict, List, Tuple
from datetime import timedelta
import functools
import math
import time
import re
import shutil
import json
from parlai.core.message import Message
from parlai.utils.strings import colorize
import parlai.utils.logging as logging
def set_namedtuple_defaults(namedtuple, default=None):
    """
    Set *all* of the fields for a given nametuple to a singular value.

    Additionally removes the default docstring for each field.
    Modifies the tuple in place, but returns it anyway.

    More info:
    https://stackoverflow.com/a/18348004

    :param namedtuple: A constructed collections.namedtuple
    :param default: The default value to set.

    :returns: the modified namedtuple
    """
    namedtuple.__new__.__defaults__ = (default,) * len(namedtuple._fields)
    for f in namedtuple._fields:
        del getattr(namedtuple, f).__doc__
    return namedtuple