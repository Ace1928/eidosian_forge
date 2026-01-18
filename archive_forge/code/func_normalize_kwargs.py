import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
def normalize_kwargs(kw, alias_mapping=None):
    """
    Helper function to normalize kwarg inputs.

    Parameters
    ----------
    kw : dict or None
        A dict of keyword arguments.  None is explicitly supported and treated
        as an empty dict, to support functions with an optional parameter of
        the form ``props=None``.

    alias_mapping : dict or Artist subclass or Artist instance, optional
        A mapping between a canonical name to a list of aliases, in order of
        precedence from lowest to highest.

        If the canonical value is not in the list it is assumed to have the
        highest priority.

        If an Artist subclass or instance is passed, use its properties alias
        mapping.

    Raises
    ------
    TypeError
        To match what Python raises if invalid arguments/keyword arguments are
        passed to a callable.
    """
    from matplotlib.artist import Artist
    if kw is None:
        return {}
    if alias_mapping is None:
        alias_mapping = {}
    elif isinstance(alias_mapping, type) and issubclass(alias_mapping, Artist) or isinstance(alias_mapping, Artist):
        alias_mapping = getattr(alias_mapping, '_alias_map', {})
    to_canonical = {alias: canonical for canonical, alias_list in alias_mapping.items() for alias in alias_list}
    canonical_to_seen = {}
    ret = {}
    for k, v in kw.items():
        canonical = to_canonical.get(k, k)
        if canonical in canonical_to_seen:
            raise TypeError(f'Got both {canonical_to_seen[canonical]!r} and {k!r}, which are aliases of one another')
        canonical_to_seen[canonical] = k
        ret[canonical] = v
    return ret