import datetime
import os
import pickle
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional
from unittest import TestCase
from uuid import uuid4
from triad.utils.io import write_text, join
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from fsspec.implementations.local import LocalFileSystem
from pytest import raises
from triad import SerializableRLock
import fugue.api as fa
from fugue import (
from fugue.column import col
from fugue.column import functions as ff
from fugue.column import lit
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import (
@cotransformer(Schema('a:int,ct1:int,p:int'))
def mock_co_tf3(df1: List[Dict[str, Any]], p=1) -> List[List[Any]]:
    return [[df1[0]['a'], len(df1), p]]