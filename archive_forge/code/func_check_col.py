import datetime
from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
def check_col(key, name, size):
    assert getattr(store.get_storer(key).table.description, name).itemsize == size