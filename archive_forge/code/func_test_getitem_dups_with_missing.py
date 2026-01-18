from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_getitem_dups_with_missing(indexer_sl):
    ser = Series([1, 2, 3, 4], ['foo', 'bar', 'foo', 'bah'])
    with pytest.raises(KeyError, match=re.escape("['bam'] not in index")):
        indexer_sl(ser)[['foo', 'bar', 'bah', 'bam']]