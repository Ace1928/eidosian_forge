import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_json_unpack_compat():
    """Test reading old json with serialized measurements array."""
    old_json = '\n        {\n            "cirq_type": "ResultDict",\n            "params": {\n                "cirq_type": "ParamResolver",\n                "param_dict": []\n            },\n            "measurements": {\n                "m": {\n                    "packed_digits": "d32a",\n                    "binary": true,\n                    "dtype": "bool",\n                    "shape": [\n                        3,\n                        5\n                    ]\n                }\n            }\n        }\n    '
    result = cirq.read_json(json_text=old_json)
    assert result == cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'m': np.array([[True, True, False, True, False], [False, True, True, False, False], [True, False, True, False, True]], dtype=bool)})