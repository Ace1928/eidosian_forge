import numpy as np
from pygsp import filters, utils
@utils.filterbank_handler
def compute_tig(g, **kwargs):
    """
    Compute the Tig for a given filter or filter bank.

    .. math:: T_ig(n) = g(L)_{i, n}

    Parameters
    ----------
    g: Filter
        One of :mod:`pygsp.filters`.
    kwargs: dict
        Additional parameters to be passed to the
        :func:`pygsp.filters.Filter.filter` method.
    """
    return g.compute_frame()