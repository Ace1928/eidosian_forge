import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.skipif(using_pyarrow_string_dtype(), reason='Adjust expected when infer_string is default, no bug here, just a complicated parametrization')
@pytest.mark.parametrize('orient,expected', [('split', '{\n    "columns":[\n        "a",\n        "b"\n    ],\n    "index":[\n        0,\n        1\n    ],\n    "data":[\n        [\n            "foo",\n            "bar"\n        ],\n        [\n            "baz",\n            "qux"\n        ]\n    ]\n}'), ('records', '[\n    {\n        "a":"foo",\n        "b":"bar"\n    },\n    {\n        "a":"baz",\n        "b":"qux"\n    }\n]'), ('index', '{\n    "0":{\n        "a":"foo",\n        "b":"bar"\n    },\n    "1":{\n        "a":"baz",\n        "b":"qux"\n    }\n}'), ('columns', '{\n    "a":{\n        "0":"foo",\n        "1":"baz"\n    },\n    "b":{\n        "0":"bar",\n        "1":"qux"\n    }\n}'), ('values', '[\n    [\n        "foo",\n        "bar"\n    ],\n    [\n        "baz",\n        "qux"\n    ]\n]'), ('table', '{\n    "schema":{\n        "fields":[\n            {\n                "name":"index",\n                "type":"integer"\n            },\n            {\n                "name":"a",\n                "type":"string"\n            },\n            {\n                "name":"b",\n                "type":"string"\n            }\n        ],\n        "primaryKey":[\n            "index"\n        ],\n        "pandas_version":"1.4.0"\n    },\n    "data":[\n        {\n            "index":0,\n            "a":"foo",\n            "b":"bar"\n        },\n        {\n            "index":1,\n            "a":"baz",\n            "b":"qux"\n        }\n    ]\n}')])
def test_json_indent_all_orients(self, orient, expected):
    df = DataFrame([['foo', 'bar'], ['baz', 'qux']], columns=['a', 'b'])
    result = df.to_json(orient=orient, indent=4)
    assert result == expected