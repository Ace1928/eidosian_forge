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
@consolidate_datasets.setter
def consolidate_datasets(self, value: bool) -> None:
    self._global_settings['consolidate_datasets'] = value