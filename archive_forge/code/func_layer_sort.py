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
def layer_sort(hmap):
    """
   Find a global ordering for layers in a HoloMap of CompositeOverlay
   types.
   """
    orderings = {}
    for o in hmap:
        okeys = [get_overlay_spec(o, k, v) for k, v in o.data.items()]
        if len(okeys) == 1 and okeys[0] not in orderings:
            orderings[okeys[0]] = []
        else:
            orderings.update({k: [] if k == v else [v] for k, v in zip(okeys[1:], okeys)})
    return [i for g in sort_topologically(orderings) for i in sorted(g)]