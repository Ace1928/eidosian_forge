from functools import wraps
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import freq_to_period
def pandas_wrapper_freq(func, trim_head=None, trim_tail=None, freq_kw='freq', columns=None, *args, **kwargs):
    """
    Return a new function that catches the incoming X, checks if it's pandas,
    calls the functions as is. Then wraps the results in the incoming index.

    Deals with frequencies. Expects that the function returns a tuple,
    a Bunch object, or a pandas-object.
    """

    @wraps(func)
    def new_func(X, *args, **kwargs):
        if not _is_using_pandas(X, None):
            return func(X, *args, **kwargs)
        wrapper_func = _get_pandas_wrapper(X, trim_head, trim_tail, columns)
        index = X.index
        freq = index.inferred_freq
        kwargs.update({freq_kw: freq_to_period(freq)})
        ret = func(X, *args, **kwargs)
        ret = wrapper_func(ret)
        return ret
    return new_func