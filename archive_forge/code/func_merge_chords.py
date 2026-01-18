import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
def merge_chords(chords):
    """
    Merge consecutive chord annotations if they represent the same chord.

    Parameters
    ----------
    chords : numpy structured arrray
        Chord annotations to be merged, in `CHORD_ANN_DTYPE` format.

    Returns
    -------
    merged_chords : numpy structured array
        Merged chord annotations, in `CHORD_ANN_DTYPE` format.

    """
    merged_starts = []
    merged_ends = []
    merged_chords = []
    prev_chord = None
    for start, end, chord in chords:
        if chord != prev_chord:
            prev_chord = chord
            merged_starts.append(start)
            merged_ends.append(end)
            merged_chords.append(chord)
        else:
            merged_ends[-1] = end
    crds = np.zeros(len(merged_chords), dtype=CHORD_ANN_DTYPE)
    crds['start'] = merged_starts
    crds['end'] = merged_ends
    crds['chord'] = merged_chords
    return crds