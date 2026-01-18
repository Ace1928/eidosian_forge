import collections
import copy
import datetime as dt
import glob
import inspect
import numbers
import os.path
import pathlib
import re
import sys
import typing
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from .parameterized import (
from ._utils import (
def param_union(*parameterizeds, warn=True):
    """
    Given a set of Parameterized objects, returns a dictionary
    with the union of all param name,value pairs across them.

    Parameters
    ----------
    warn : bool, optional
        Wether to warn if the same parameter have been given multiple values,
        otherwise use the last value, by default True

    Returns
    -------
    dict
        Union of all param name,value pairs
    """
    d = {}
    for o in parameterizeds:
        for k in o.param:
            if k != 'name':
                if k in d and warn:
                    get_logger().warning(f'overwriting parameter {k}')
                d[k] = getattr(o, k)
    return d