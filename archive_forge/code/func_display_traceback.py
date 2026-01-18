from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import json
import itertools
import re
import sys
import traceback
import warnings
from typing import (
from types import ModuleType
import jsonschema
import pandas as pd
import numpy as np
from pandas.api.types import infer_dtype
from altair.utils.schemapi import SchemaBase
from altair.utils._dfi_types import Column, DtypeKind, DataFrame as DfiDataFrame
from typing import Literal, Protocol, TYPE_CHECKING, runtime_checkable
def display_traceback(in_ipython: bool=True):
    exc_info = sys.exc_info()
    if in_ipython:
        from IPython.core.getipython import get_ipython
        ip = get_ipython()
    else:
        ip = None
    if ip is not None:
        ip.showtraceback(exc_info)
    else:
        traceback.print_exception(*exc_info)