from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
@pytest.fixture
def expected_without_index(self, datapath):
    return expected_html(datapath, 'index_2')