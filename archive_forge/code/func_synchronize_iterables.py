import os
import sys
import pickle
from collections import defaultdict
import re
from copy import deepcopy
from glob import glob
from pathlib import Path
from traceback import format_exception
from hashlib import sha1
from functools import reduce
import numpy as np
from ... import logging, config
from ...utils.filemanip import (
from ...utils.misc import str2bool
from ...utils.functions import create_function_from_source
from ...interfaces.base.traits_extension import (
from ...interfaces.base.support import Bunch, InterfaceResult
from ...interfaces.base import CommandLine
from ...interfaces.utility import IdentityInterface
from ...utils.provenance import ProvStore, pm, nipype_ns, get_id
from inspect import signature
def synchronize_iterables(iterables):
    """Synchronize the given iterables in item-wise order.

    Return: the {field: value} dictionary list

    Examples
    --------
    >>> from nipype.pipeline.engine.utils import synchronize_iterables
    >>> iterables = dict(a=lambda: [1, 2], b=lambda: [3, 4])
    >>> synced = synchronize_iterables(iterables)
    >>> synced == [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]
    True
    >>> iterables = dict(a=lambda: [1, 2], b=lambda: [3], c=lambda: [4, 5, 6])
    >>> synced = synchronize_iterables(iterables)
    >>> synced == [{'a': 1, 'b': 3, 'c': 4}, {'a': 2, 'c': 5}, {'c': 6}]
    True
    """
    out_list = []
    iterable_items = [(field, iter(fvals())) for field, fvals in sorted(iterables.items())]
    while True:
        cur_dict = {}
        for field, iter_values in iterable_items:
            try:
                cur_dict[field] = next(iter_values)
            except StopIteration:
                pass
        if cur_dict:
            out_list.append(cur_dict)
        else:
            break
    return out_list