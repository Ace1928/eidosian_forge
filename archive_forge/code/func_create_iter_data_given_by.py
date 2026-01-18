from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.missing import remove_na_arraylike
from pandas import (
from pandas.plotting._matplotlib.misc import unpack_single_str_list
def create_iter_data_given_by(data: DataFrame, kind: str='hist') -> dict[Hashable, DataFrame | Series]:
    """
    Create data for iteration given `by` is assigned or not, and it is only
    used in both hist and boxplot.

    If `by` is assigned, return a dictionary of DataFrames in which the key of
    dictionary is the values in groups.
    If `by` is not assigned, return input as is, and this preserves current
    status of iter_data.

    Parameters
    ----------
    data : reformatted grouped data from `_compute_plot_data` method.
    kind : str, plot kind. This function is only used for `hist` and `box` plots.

    Returns
    -------
    iter_data : DataFrame or Dictionary of DataFrames

    Examples
    --------
    If `by` is assigned:

    >>> import numpy as np
    >>> tuples = [('h1', 'a'), ('h1', 'b'), ('h2', 'a'), ('h2', 'b')]
    >>> mi = pd.MultiIndex.from_tuples(tuples)
    >>> value = [[1, 3, np.nan, np.nan],
    ...          [3, 4, np.nan, np.nan], [np.nan, np.nan, 5, 6]]
    >>> data = pd.DataFrame(value, columns=mi)
    >>> create_iter_data_given_by(data)
    {'h1':     h1
         a    b
    0  1.0  3.0
    1  3.0  4.0
    2  NaN  NaN, 'h2':     h2
         a    b
    0  NaN  NaN
    1  NaN  NaN
    2  5.0  6.0}
    """
    if kind == 'hist':
        level = 0
    else:
        level = 1
    assert isinstance(data.columns, MultiIndex)
    return {col: data.loc[:, data.columns.get_level_values(level) == col] for col in data.columns.levels[level]}