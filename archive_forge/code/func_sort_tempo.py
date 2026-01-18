from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from . import EvaluationMixin, MeanEvaluation, evaluation_io
from ..io import load_tempo
def sort_tempo(tempo):
    """
    Sort tempi according to their strengths.

    Parameters
    ----------
    tempo : numpy array, shape (num_tempi, 2)
        Tempi (first column) and their relative strength (second column).

    Returns
    -------
    tempi : numpy array, shape (num_tempi, 2)
        Tempi sorted according to their strength.

    """
    tempo = np.array(tempo, copy=False, ndmin=1)
    if tempo.ndim != 2:
        raise ValueError('`tempo` has no strength information, cannot sort them.')
    tempi = tempo[:, 0]
    strengths = tempo[:, 1]
    sort_idx = (-strengths).argsort(kind='mergesort')
    tempi = tempi[sort_idx]
    strengths = strengths[sort_idx]
    return np.vstack((tempi, strengths)).T