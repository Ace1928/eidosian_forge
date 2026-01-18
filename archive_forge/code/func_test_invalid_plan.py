import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
def test_invalid_plan():
    query = '\n    {\n        "relations": [\n        ]\n    }\n    '
    buf = pa._substrait._parse_json_plan(tobytes(query))
    exec_message = 'Plan has no relations'
    with pytest.raises(ArrowInvalid, match=exec_message):
        substrait.run_query(buf)