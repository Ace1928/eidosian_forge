from functools import partial
import gzip
from io import BytesIO
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def html_responder(df):
    return df.to_html(index=False).encode('utf-8')