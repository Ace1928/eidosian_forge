import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('levels1, levels2, codes1, codes2, names', [([['a', 'b', 'c'], [0, '']], [['c', 'd', 'b'], ['']], [[0, 1, 2], [1, 1, 1]], [[0, 1, 2], [0, 0, 0]], ['name1', 'name2'])])
def test_intersection_lexsort_depth(levels1, levels2, codes1, codes2, names):
    mi1 = MultiIndex(levels=levels1, codes=codes1, names=names)
    mi2 = MultiIndex(levels=levels2, codes=codes2, names=names)
    mi_int = mi1.intersection(mi2)
    assert mi_int._lexsort_depth == 2