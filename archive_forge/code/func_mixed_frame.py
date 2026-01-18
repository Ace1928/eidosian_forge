import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
@pytest.fixture
def mixed_frame():
    return DataFrame({'a': [1, 2, 3, 4], 'b': [1.0, 3.0, 27.0, 81.0], 'c': ['Atlanta', 'Birmingham', 'Cincinnati', 'Detroit']})