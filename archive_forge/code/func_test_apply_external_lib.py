from __future__ import annotations
import datetime
import itertools
import json
import unittest.mock as mock
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
from pandas.core.indexing import IndexingError
from pandas.errors import SpecificationError
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_series_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution, try_cast_to_pandas
from .utils import (
def test_apply_external_lib():
    json_string = '\n    {\n        "researcher": {\n            "name": "Ford Prefect",\n            "species": "Betelgeusian",\n            "relatives": [\n                {\n                    "name": "Zaphod Beeblebrox",\n                    "species": "Betelgeusian"\n                }\n            ]\n        }\n    }\n    '
    modin_result = pd.DataFrame.from_dict({'a': [json_string]}).a.apply(json.loads)
    pandas_result = pandas.DataFrame.from_dict({'a': [json_string]}).a.apply(json.loads)
    df_equals(modin_result, pandas_result)