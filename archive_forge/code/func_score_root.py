import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
def score_root(det_chords, ann_chords):
    """
    Score similarity of chords based on only the root, i.e. returns a score of
    1 if roots match, 0 otherwise.

    Parameters
    ----------
    det_chords : numpy structured array
        Detected chords.
    ann_chords : numpy structured array
        Annotated chords.

    Returns
    -------
    scores : numpy array
        Similarity score for each chord.

    """
    return (ann_chords['root'] == det_chords['root']).astype(np.float)