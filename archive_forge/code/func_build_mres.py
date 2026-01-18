import os
from io import open
import types
from functools import wraps, partial
from contextlib import contextmanager
import sys, re
import sre_parse
import sre_constants
from inspect import getmembers, getmro
from functools import partial, wraps
from itertools import repeat, product
def build_mres(terminals, g_regex_flags, match_whole=False):
    return _build_mres(terminals, len(terminals), g_regex_flags, match_whole)