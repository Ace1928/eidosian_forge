import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
def skew_reduce(dfgb, *args, **kwargs):
    df = dfgb.sum(*args, **kwargs)
    if df.empty:
        return df.droplevel(GroupByReduce.ID_LEVEL_NAME, axis=1)
    count = df['count']
    s = df['sum']
    s2 = df['pow2_sum']
    s3 = df['pow3_sum']
    m = s / count
    m2 = s2 - 2 * m * s + count * m ** 2
    m3 = s3 - 3 * m * s2 + 3 * s * m ** 2 - count * m ** 3
    with np.errstate(invalid='ignore', divide='ignore'):
        skew_res = count * (count - 1) ** 0.5 / (count - 2) * (m3 / m2 ** 1.5)
    skew_res[m2 == 0] = 0
    skew_res[count < 3] = np.nan
    return skew_res