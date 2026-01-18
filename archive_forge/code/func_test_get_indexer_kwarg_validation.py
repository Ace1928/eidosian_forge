from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer_kwarg_validation(self):
    mi = MultiIndex.from_product([range(3), ['A', 'B']])
    msg = 'limit argument only valid if doing pad, backfill or nearest'
    with pytest.raises(ValueError, match=msg):
        mi.get_indexer(mi[:-1], limit=4)
    msg = 'tolerance argument only valid if doing pad, backfill or nearest'
    with pytest.raises(ValueError, match=msg):
        mi.get_indexer(mi[:-1], tolerance='piano')