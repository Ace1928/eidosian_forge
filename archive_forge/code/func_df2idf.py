import logging
from functools import partial
import re
import numpy as np
from gensim import interfaces, matutils, utils
from gensim.utils import deprecated
def df2idf(docfreq, totaldocs, log_base=2.0, add=0.0):
    """Compute inverse-document-frequency for a term with the given document frequency `docfreq`:
    :math:`idf = add + log_{log\\_base} \\frac{totaldocs}{docfreq}`

    Parameters
    ----------
    docfreq : {int, float}
        Document frequency.
    totaldocs : int
        Total number of documents.
    log_base : float, optional
        Base of logarithm.
    add : float, optional
        Offset.

    Returns
    -------
    float
        Inverse document frequency.

    """
    return add + np.log(float(totaldocs) / docfreq) / np.log(log_base)