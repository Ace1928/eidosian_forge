from numpy.testing import assert_equal
import numpy as np
def dummy_1d(x, varname=None):
    """dummy variable for id integer groups

    Parameters
    ----------
    x : ndarray, 1d
        categorical variable, requires integers if varname is None
    varname : str
        name of the variable used in labels for category levels

    Returns
    -------
    dummy : ndarray, 2d
        array of dummy variables, one column for each level of the
        category (full set)
    labels : list[str]
        labels for the columns, i.e. levels of each category


    Notes
    -----
    use tools.categorical instead for more more options

    See Also
    --------
    statsmodels.tools.categorical

    Examples
    --------
    >>> x = np.array(['F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M'],
          dtype='|S1')
    >>> dummy_1d(x, varname='gender')
    (array([[1, 0],
           [1, 0],
           [0, 1],
           [0, 1],
           [1, 0],
           [1, 0],
           [0, 1],
           [0, 1],
           [1, 0],
           [1, 0],
           [0, 1],
           [0, 1]]), ['gender_F', 'gender_M'])

    """
    if varname is None:
        labels = ['level_%d' % i for i in range(x.max() + 1)]
        return ((x[:, None] == np.arange(x.max() + 1)).astype(int), labels)
    else:
        grouplabels = np.unique(x)
        labels = [varname + '_%s' % str(i) for i in grouplabels]
        return ((x[:, None] == grouplabels).astype(int), labels)