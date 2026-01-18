from __future__ import annotations
import ast
import base64
import datetime as dt
import json
import logging
import numbers
import os
import pathlib
import re
import sys
import urllib.parse as urlparse
from collections import OrderedDict, defaultdict
from collections.abc import MutableMapping, MutableSequence
from datetime import datetime
from functools import partial
from html import escape  # noqa
from importlib import import_module
from typing import Any, AnyStr
import bokeh
import numpy as np
import param
from bokeh.core.has_props import _default_resolver
from bokeh.model import Model
from packaging.version import Version
from .checks import (  # noqa
from .parameters import (  # noqa
def styler_update(styler, new_df):
    """
    Updates the todo items on a pandas Styler object to apply to a new
    DataFrame.

    Arguments
    ---------
    styler: pandas.io.formats.style.Styler
      Styler objects
    new_df: pd.DataFrame
      New DataFrame to update the styler to do items

    Returns
    -------
    todos: list
    """
    todos = []
    for todo in styler._todo:
        if not isinstance(todo, tuple):
            todos.append(todo)
            continue
        ops = []
        for op in todo:
            if not isinstance(op, tuple):
                ops.append(op)
                continue
            op_fn = str(op[0])
            if ('_background_gradient' in op_fn or '_bar' in op_fn) and op[1] in (0, 1):
                if isinstance(op[2], list):
                    applies = op[2]
                else:
                    applies = np.array([new_df[col].dtype.kind in 'uif' for col in new_df.columns])
                    if len(op[2]) == len(applies):
                        applies = np.logical_and(applies, op[2])
                op = (op[0], op[1], applies)
            ops.append(op)
        todo = tuple(ops)
        todos.append(todo)
    return todos