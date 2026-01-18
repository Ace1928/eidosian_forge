import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
@classmethod
def build_qc_method(cls, agg_name, finalizer_fn=None):
    """
        Build a TreeReduce implemented query compiler method for the specified groupby aggregation.

        Parameters
        ----------
        agg_name : hashable
        finalizer_fn : callable(pandas.DataFrame) -> pandas.DataFrame, default: None
            A callable to execute at the end a groupby kernel against groupby result.

        Returns
        -------
        callable
            Function that takes query compiler and executes GroupBy aggregation
            with TreeReduce algorithm.
        """
    map_fn, reduce_fn, d2p_fn = cls.get_impl(agg_name)
    map_reduce_method = GroupByReduce.register(map_fn, reduce_fn, default_to_pandas_func=d2p_fn, finalizer_fn=finalizer_fn)

    def method(query_compiler, *args, **kwargs):
        if use_range_partitioning_groupby():
            try:
                if finalizer_fn is not None:
                    raise NotImplementedError('Range-partitioning groupby is not implemented yet when a finalizing function is specified.')
                return query_compiler._groupby_shuffle(*args, agg_func=agg_name, **kwargs)
            except NotImplementedError as e:
                ErrorMessage.warn(f"Can't use range-partitioning groupby implementation because of: {e}" + '\nFalling back to a TreeReduce implementation.')
        return map_reduce_method(query_compiler, *args, **kwargs)
    return method