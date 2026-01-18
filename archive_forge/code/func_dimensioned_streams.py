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
def dimensioned_streams(dmap):
    """
    Given a DynamicMap return all streams that have any dimensioned
    parameters, i.e. parameters also listed in the key dimensions.
    """
    dimensioned = []
    for stream in dmap.streams:
        stream_params = stream_parameters([stream])
        if {str(k) for k in dmap.kdims} & set(stream_params):
            dimensioned.append(stream)
    return dimensioned