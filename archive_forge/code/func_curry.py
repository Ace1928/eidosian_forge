import json
import os
import random
import hashlib
import warnings
from typing import Union, MutableMapping, Optional, Dict, Sequence, TYPE_CHECKING, List
import pandas as pd
from toolz import curried
from typing import TypeVar
from ._importers import import_pyarrow_interchange
from .core import sanitize_dataframe, sanitize_arrow_table, DataFrameLike
from .core import sanitize_geo_interface
from .deprecation import AltairDeprecationWarning
from .plugin_registry import PluginRegistry
from typing import Protocol, TypedDict, Literal
def curry(*args, **kwargs):
    """Curry a callable function

    Deprecated: use toolz.curried.curry() instead.
    """
    warnings.warn('alt.curry() is deprecated, and will be removed in a future release. Use toolz.curried.curry() instead.', AltairDeprecationWarning, stacklevel=1)
    return curried.curry(*args, **kwargs)