import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def multiindex_frame(self):
    """Multiindex dataframe for testing multirow LaTeX macros."""
    yield DataFrame.from_dict({('c1', 0): Series({x: x for x in range(4)}), ('c1', 1): Series({x: x + 4 for x in range(4)}), ('c2', 0): Series({x: x for x in range(4)}), ('c2', 1): Series({x: x + 4 for x in range(4)}), ('c3', 0): Series({x: x for x in range(4)})}).T