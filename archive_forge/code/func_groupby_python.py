import builtins
import datetime as dt
import hashlib
import inspect
import itertools
import json
import numbers
import operator
import pickle
import string
import sys
import time
import types
import unicodedata
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import partial
from threading import Event, Thread
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
@param.parameterized.bothmethod
def groupby_python(self_or_cls, ndmapping, dimensions, container_type, group_type, sort=False, **kwargs):
    idims = [dim for dim in ndmapping.kdims if dim not in dimensions]
    dim_names = [dim.name for dim in dimensions]
    selects = get_unique_keys(ndmapping, dimensions)
    selects = group_select(list(selects))
    groups = [(k, group_type(v.reindex(idims) if hasattr(v, 'kdims') else [((), v)], **kwargs)) for k, v in iterative_select(ndmapping, dim_names, selects)]
    return container_type(groups, kdims=dimensions)