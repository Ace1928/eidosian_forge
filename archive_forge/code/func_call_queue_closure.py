import warnings
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.python.common import PythonWrapper
def call_queue_closure(data, call_queue):
    """
            Apply callables from `call_queue` on copy of the `data` and return the result.

            Parameters
            ----------
            data : pandas.DataFrame or pandas.Series
                Data to use for computations.
            call_queue : array-like
                Array with callables and it's kwargs to be applied to the `data`.

            Returns
            -------
            pandas.DataFrame or pandas.Series
            """
    result = data.copy()
    for func, f_args, f_kwargs in call_queue:
        try:
            result = func(result, *f_args, **f_kwargs)
        except Exception as err:
            self.call_queue = []
            raise err
    return result