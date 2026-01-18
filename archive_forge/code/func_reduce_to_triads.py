import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
def reduce_to_triads(chords, keep_bass=False):
    """
    Reduce chords to triads.

    The function follows the reduction rules implemented in [1]_. If a chord
    chord does not contain a third, major second or fourth, it is reduced to
    a power chord. If it does not contain neither a third nor a fifth, it is
    reduced to a single note "chord".

    Parameters
    ----------
    chords : numpy structured array
        Chords to be reduced.
    keep_bass : bool
        Indicates whether to keep the bass note or set it to 0.

    Returns
    -------
    reduced_chords : numpy structured array
        Chords reduced to triads.

    References
    ----------
    .. [1] Johan Pauwels and Geoffroy Peeters.
           "Evaluating Automatically Estimated Chord Sequences."
           In Proceedings of ICASSP 2013, Vancouver, Canada, 2013.

    """
    unison = chords['intervals'][:, 0].astype(bool)
    maj_sec = chords['intervals'][:, 2].astype(bool)
    min_third = chords['intervals'][:, 3].astype(bool)
    maj_third = chords['intervals'][:, 4].astype(bool)
    perf_fourth = chords['intervals'][:, 5].astype(bool)
    dim_fifth = chords['intervals'][:, 6].astype(bool)
    perf_fifth = chords['intervals'][:, 7].astype(bool)
    aug_fifth = chords['intervals'][:, 8].astype(bool)
    no_chord = (chords['intervals'] == NO_CHORD[-1]).all(axis=1)
    reduced_chords = chords.copy()
    ivs = reduced_chords['intervals']
    ivs[~no_chord] = interval_list('(1)')
    ivs[unison & perf_fifth] = interval_list('(1,5)')
    ivs[~perf_fourth & maj_sec] = _shorthands['sus2']
    ivs[perf_fourth & ~maj_sec] = _shorthands['sus4']
    ivs[min_third] = _shorthands['min']
    ivs[min_third & aug_fifth & ~perf_fifth] = interval_list('(1,b3,#5)')
    ivs[min_third & dim_fifth & ~perf_fifth] = _shorthands['dim']
    ivs[maj_third] = _shorthands['maj']
    ivs[maj_third & dim_fifth & ~perf_fifth] = interval_list('(1,3,b5)')
    ivs[maj_third & aug_fifth & ~perf_fifth] = _shorthands['aug']
    if not keep_bass:
        reduced_chords['bass'] = 0
    else:
        reduced_chords['bass'] *= ivs[range(len(reduced_chords)), reduced_chords['bass']]
    reduced_chords['bass'][no_chord] = -1
    return reduced_chords