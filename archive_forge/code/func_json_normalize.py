from __future__ import annotations
import csv
import inspect
import pathlib
import pickle
import warnings
from typing import (
import numpy as np
import pandas
from pandas._libs.lib import NoDefault, no_default
from pandas._typing import (
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.readers import _c_parser_defaults
from modin.config import ExperimentalNumPyAPI
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, enable_logging
from modin.utils import (
@_inherit_docstrings(pandas.json_normalize, apilink='pandas.json_normalize')
@enable_logging
def json_normalize(data: Union[Dict, List[Dict]], record_path: Optional[Union[str, List]]=None, meta: Optional[Union[str, List[Union[str, List[str]]]]]=None, meta_prefix: Optional[str]=None, record_prefix: Optional[str]=None, errors: Optional[str]='raise', sep: str='.', max_level: Optional[int]=None) -> DataFrame:
    """
    Normalize semi-structured JSON data into a flat table.
    """
    ErrorMessage.default_to_pandas('json_normalize')
    return ModinObjects.DataFrame(pandas.json_normalize(data, record_path, meta, meta_prefix, record_prefix, errors, sep, max_level))