import pandas
from modin.core.dataframe.pandas.metadata import ModinIndex
from modin.error_message import ErrorMessage
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default2pandas.groupby import GroupBy, GroupByDefault
from .tree_reduce import TreeReduce
def build_fn(name):
    return lambda df, *args, **kwargs: getattr(df, name)(*args, **kwargs)