import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
@pytest.fixture
def state_data():
    return [{'counties': [{'name': 'Dade', 'population': 12345}, {'name': 'Broward', 'population': 40000}, {'name': 'Palm Beach', 'population': 60000}], 'info': {'governor': 'Rick Scott'}, 'shortname': 'FL', 'state': 'Florida'}, {'counties': [{'name': 'Summit', 'population': 1234}, {'name': 'Cuyahoga', 'population': 1337}], 'info': {'governor': 'John Kasich'}, 'shortname': 'OH', 'state': 'Ohio'}]