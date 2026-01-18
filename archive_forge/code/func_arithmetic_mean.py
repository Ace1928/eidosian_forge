import logging
import numpy as np
def arithmetic_mean(confirmed_measures):
    """
    Perform the arithmetic mean aggregation on the output obtained from
    the confirmation measure module.

    Parameters
    ----------
    confirmed_measures : list of float
        List of calculated confirmation measure on each set in the segmented topics.

    Returns
    -------
    `numpy.float`
        Arithmetic mean of all the values contained in confirmation measures.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.topic_coherence.aggregation import arithmetic_mean
        >>> arithmetic_mean([1.1, 2.2, 3.3, 4.4])
        2.75

    """
    return np.mean(confirmed_measures)