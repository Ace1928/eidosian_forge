import io
import pytest
import pandas as pd
def test_dtype_name_in_info(self, data):
    buf = io.StringIO()
    pd.DataFrame({'A': data}).info(buf=buf)
    result = buf.getvalue()
    assert data.dtype.name in result