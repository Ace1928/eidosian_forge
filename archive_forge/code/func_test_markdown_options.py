import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
def test_markdown_options(fsspectest):
    pytest.importorskip('tabulate')
    df = DataFrame({'a': [0]})
    df.to_markdown('testmem://mockfile', storage_options={'test': 'md_write'})
    assert fsspectest.test[0] == 'md_write'
    assert fsspectest.cat('testmem://mockfile')