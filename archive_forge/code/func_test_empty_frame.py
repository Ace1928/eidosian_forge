from io import (
import pytest
import pandas as pd
import pandas._testing as tm
def test_empty_frame():
    buf = StringIO()
    df = pd.DataFrame({'id': [], 'first_name': [], 'last_name': []}).set_index('id')
    df.to_markdown(buf=buf)
    result = buf.getvalue()
    assert result == '| id   | first_name   | last_name   |\n|------|--------------|-------------|'