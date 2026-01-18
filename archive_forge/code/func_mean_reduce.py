import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
def mean_reduce(dfgb, **kwargs):
    """
            Compute mean value in each group using sums/counts values within reduce phase.

            Parameters
            ----------
            dfgb : pandas.DataFrameGroupBy
                GroupBy object for column-partition.
            **kwargs : dict
                Additional keyword parameters to be passed in ``pandas.DataFrameGroupBy.sum``.

            Returns
            -------
            pandas.DataFrame
                A pandas Dataframe with mean values in each column of each group.
            """
    sums_counts_df = dfgb.sum(**kwargs)
    if sums_counts_df.empty:
        return sums_counts_df.droplevel(GroupByReduce.ID_LEVEL_NAME, axis=1)
    sum_df = sums_counts_df['sum']
    count_df = sums_counts_df['count']
    return sum_df / count_df