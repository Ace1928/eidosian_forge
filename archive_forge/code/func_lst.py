import re
import pytest
from pandas.core.indexes.frozen import FrozenList
@pytest.fixture
def lst():
    return [1, 2, 3, 4, 5]