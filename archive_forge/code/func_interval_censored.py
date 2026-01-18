import numpy as np
@classmethod
def interval_censored(cls, low, high):
    """
        Create a `CensoredData` instance of interval-censored data.

        This method is useful when all the data is interval-censored, and
        the low and high ends of the intervals are already stored in
        separate one-dimensional arrays.

        Parameters
        ----------
        low : array_like
            The one-dimensional array containing the low ends of the
            intervals.
        high : array_like
            The one-dimensional array containing the high ends of the
            intervals.

        Returns
        -------
        data : `CensoredData`
            An instance of `CensoredData` that represents the
            collection of censored values.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import CensoredData

        ``a`` and ``b`` are the low and high ends of a collection of
        interval-censored values.

        >>> a = [0.5, 2.0, 3.0, 5.5]
        >>> b = [1.0, 2.5, 3.5, 7.0]
        >>> data = CensoredData.interval_censored(low=a, high=b)
        >>> print(data)
        CensoredData(4 values: 0 not censored, 4 interval-censored)
        """
    _validate_1d(low, 'low', allow_inf=True)
    _validate_1d(high, 'high', allow_inf=True)
    if len(low) != len(high):
        raise ValueError('`low` and `high` must have the same length.')
    interval = np.column_stack((low, high))
    uncensored, left, right, interval = _validate_interval(interval)
    return cls(uncensored=uncensored, left=left, right=right, interval=interval)