import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_donot_drop_nonevalues(self):
    data = [{'info': None, 'author_name': {'first': 'Smith', 'last_name': 'Appleseed'}}, {'info': {'created_at': '11/08/1993', 'last_updated': '26/05/2012'}, 'author_name': {'first': 'Jane', 'last_name': 'Doe'}}]
    result = nested_to_record(data)
    expected = [{'info': None, 'author_name.first': 'Smith', 'author_name.last_name': 'Appleseed'}, {'author_name.first': 'Jane', 'author_name.last_name': 'Doe', 'info.created_at': '11/08/1993', 'info.last_updated': '26/05/2012'}]
    assert result == expected