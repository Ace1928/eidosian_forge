from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def cartesian_product_for_groupers(result, args, names, fill_value=np.nan):
    """Reindex to a cartesian production for the groupers,
    preserving the nature (Categorical) of each grouper
    """

    def f(a):
        if isinstance(a, (CategoricalIndex, Categorical)):
            categories = a.categories
            a = Categorical.from_codes(np.arange(len(categories)), categories=categories, ordered=a.ordered)
        return a
    index = MultiIndex.from_product(map(f, args), names=names)
    return result.reindex(index, fill_value=fill_value).sort_index()