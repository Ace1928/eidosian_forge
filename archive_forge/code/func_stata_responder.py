from functools import partial
import gzip
from io import BytesIO
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def stata_responder(df):
    with BytesIO() as bio:
        df.to_stata(bio, write_index=False)
        return bio.getvalue()