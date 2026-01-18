from functools import partial
import gzip
from io import BytesIO
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def gz_json_responder(df):
    return gzip_bytes(json_responder(df))