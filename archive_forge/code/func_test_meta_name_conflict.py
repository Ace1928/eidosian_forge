import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_meta_name_conflict(self):
    data = [{'foo': 'hello', 'bar': 'there', 'data': [{'foo': 'something', 'bar': 'else'}, {'foo': 'something2', 'bar': 'else2'}]}]
    msg = 'Conflicting metadata name (foo|bar), need distinguishing prefix'
    with pytest.raises(ValueError, match=msg):
        json_normalize(data, 'data', meta=['foo', 'bar'])
    result = json_normalize(data, 'data', meta=['foo', 'bar'], meta_prefix='meta')
    for val in ['metafoo', 'metabar', 'foo', 'bar']:
        assert val in result