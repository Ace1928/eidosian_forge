import numpy as np
@classmethod
def right_censored(cls, x, censored):
    """
        Create a `CensoredData` instance of right-censored data.

        Parameters
        ----------
        x : array_like
            `x` is the array of observed data or measurements.
            `x` must be a one-dimensional sequence of finite numbers.
        censored : array_like of bool
            `censored` must be a one-dimensional sequence of boolean
            values.  If ``censored[k]`` is True, the corresponding value
            in `x` is right-censored.  That is, the value ``x[k]``
            is the lower bound of the true (but unknown) value.

        Returns
        -------
        data : `CensoredData`
            An instance of `CensoredData` that represents the
            collection of uncensored and right-censored values.

        Examples
        --------
        >>> from scipy.stats import CensoredData

        Two uncensored values (4 and 10) and two right-censored values
        (24 and 25).

        >>> data = CensoredData.right_censored([4, 10, 24, 25],
        ...                                    [False, False, True, True])
        >>> data
        CensoredData(uncensored=array([ 4., 10.]),
        left=array([], dtype=float64), right=array([24., 25.]),
        interval=array([], shape=(0, 2), dtype=float64))
        >>> print(data)
        CensoredData(4 values: 2 not censored, 2 right-censored)
        """
    x, censored = _validate_x_censored(x, censored)
    return cls(uncensored=x[~censored], right=x[censored])