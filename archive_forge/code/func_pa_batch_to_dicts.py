import json
import pickle
from datetime import date, datetime
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import io
import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version
from pandas.core.dtypes.base import ExtensionDtype
from pyarrow.compute import CastOptions, binary_join_element_wise
from pyarrow.json import read_json, ParseOptions as JsonParseOptions
from triad.constants import TRIAD_VAR_QUOTE
from .convert import as_type
from .iter import EmptyAwareIterable, Slicer
from .json import loads_no_dup
from .schema import move_to_unquoted, quote_name, unquote_name
from .assertion import assert_or_throw
def pa_batch_to_dicts(batch: pa.RecordBatch) -> List[Dict[str, Any]]:
    """Convert a pyarrow record batch to list of dict

    :param batch: the pyarrow record batch
    :return: the list of dict
    """
    if PYARROW_VERSION.major < 7:
        names = batch.schema.names
        return [dict(zip(names, tp)) for tp in zip(*batch.to_pydict().values())]
    else:
        return batch.to_pylist()