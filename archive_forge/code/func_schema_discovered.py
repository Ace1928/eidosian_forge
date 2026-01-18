import json
from abc import abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import pandas as pd
import pyarrow as pa
from triad import SerializableRLock
from triad.collections.schema import Schema
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import cast_pa_table
from .._utils.display import PrettyTable
from ..collections.yielded import Yielded
from ..dataset import (
from ..exceptions import FugueDataFrameOperationError
@property
def schema_discovered(self) -> Schema:
    """Whether the schema has been discovered or still a lambda"""
    return self._schema_discovered