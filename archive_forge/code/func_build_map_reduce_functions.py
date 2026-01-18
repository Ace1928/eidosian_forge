import pandas
from modin.core.dataframe.pandas.metadata import ModinIndex
from modin.error_message import ErrorMessage
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default2pandas.groupby import GroupBy, GroupByDefault
from .tree_reduce import TreeReduce
@classmethod
def build_map_reduce_functions(cls, by, axis, groupby_kwargs, map_func, reduce_func, agg_args, agg_kwargs, drop=False, method=None, finalizer_fn=None):
    """
        Bind appropriate arguments to map and reduce functions.

        Parameters
        ----------
        by : BaseQueryCompiler, column or index label, Grouper or list of such
            Object that determine groups.
        axis : {0, 1}
            Axis to group and apply aggregation function along. 0 means index axis
            when 1 means column axis.
        groupby_kwargs : dict
            Dictionary which carries arguments for pandas.DataFrame.groupby.
        map_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Map phase.
        reduce_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Reduce phase.
        agg_args : list-like
            Positional arguments to pass to the aggregation functions.
        agg_kwargs : dict
            Keyword arguments to pass to the aggregation functions.
        drop : bool, default: False
            Indicates whether or not by-data came from the `self` frame.
        method : str, optional
            Name of the GroupBy aggregation function. This is a hint to be able to do special casing.
        finalizer_fn : callable(pandas.DataFrame) -> pandas.DataFrame, default: None
            A callable to execute at the end a groupby kernel against groupby result.

        Returns
        -------
        Tuple of callable
            Tuple of map and reduce functions with bound arguments.
        """
    if hasattr(by, '_modin_frame'):
        by = None

    def _map(df, other=None, **kwargs):

        def wrapper(df, other=None):
            return cls.map(df, other=other, axis=axis, by=by, groupby_kwargs=groupby_kwargs.copy(), map_func=map_func, agg_args=agg_args, agg_kwargs=agg_kwargs, drop=drop, **kwargs)
        try:
            result = wrapper(df, other)
        except ValueError:
            result = wrapper(df.copy(), other if other is None else other.copy())
        return result

    def _reduce(df, **call_kwargs):

        def wrapper(df):
            return cls.reduce(df, axis=axis, groupby_kwargs=groupby_kwargs, reduce_func=reduce_func, agg_args=agg_args, agg_kwargs=agg_kwargs, drop=drop, method=method, finalizer_fn=finalizer_fn, **call_kwargs)
        try:
            result = wrapper(df)
        except ValueError:
            result = wrapper(df.copy())
        return result
    return (_map, _reduce)