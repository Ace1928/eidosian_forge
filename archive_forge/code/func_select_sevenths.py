import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
def select_sevenths(chords):
    """
    Compute a mask that selects all major, minor, seventh, and
    "no chords" with a 1, and all other chords with a 0.

    Parameters
    ----------
    chords : numpy structured array
        Chords to compute the mask for.

    Returns
    -------
    mask : numpy array (boolean)
        Selection mask for major, minor, seventh, and "no chords".

    """
    return select_majmin(chords) | (chords['intervals'] == _shorthands['7']).all(axis=1) | (chords['intervals'] == _shorthands['min7']).all(axis=1) | (chords['intervals'] == _shorthands['maj7']).all(axis=1)