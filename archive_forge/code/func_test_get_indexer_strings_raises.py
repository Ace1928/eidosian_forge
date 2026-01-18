from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.missing import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import Index
import pandas._testing as tm
def test_get_indexer_strings_raises(self, using_infer_string):
    index = Index(['b', 'c'])
    if using_infer_string:
        import pyarrow as pa
        msg = 'has no kernel'
        with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
            index.get_indexer(['a', 'b', 'c', 'd'], method='nearest')
        with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
            index.get_indexer(['a', 'b', 'c', 'd'], method='pad', tolerance=2)
        with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
            index.get_indexer(['a', 'b', 'c', 'd'], method='pad', tolerance=[2, 2, 2, 2])
    else:
        msg = "unsupported operand type\\(s\\) for -: 'str' and 'str'"
        with pytest.raises(TypeError, match=msg):
            index.get_indexer(['a', 'b', 'c', 'd'], method='nearest')
        with pytest.raises(TypeError, match=msg):
            index.get_indexer(['a', 'b', 'c', 'd'], method='pad', tolerance=2)
        with pytest.raises(TypeError, match=msg):
            index.get_indexer(['a', 'b', 'c', 'd'], method='pad', tolerance=[2, 2, 2, 2])