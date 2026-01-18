import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
def skew_map(dfgb, *args, **kwargs):
    if dfgb._selection is not None:
        data_to_agg = dfgb._selected_obj
    else:
        cols_to_agg = dfgb.obj.columns.difference(dfgb.exclusions)
        data_to_agg = dfgb.obj[cols_to_agg]
    df_pow2 = data_to_agg ** 2
    df_pow3 = data_to_agg ** 3
    return pandas.concat([dfgb.count(*args, **kwargs), dfgb.sum(*args, **kwargs), df_pow2.groupby(dfgb.grouper).sum(*args, **kwargs), df_pow3.groupby(dfgb.grouper).sum(*args, **kwargs)], copy=False, axis=1, keys=['count', 'sum', 'pow2_sum', 'pow3_sum'], names=[GroupByReduce.ID_LEVEL_NAME])