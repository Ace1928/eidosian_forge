import numpy as np
from collections import namedtuple
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality import topology_scaling
from ase.geometry.dimensionality.bond_generator import next_bond
def merge_intervals(intervals):
    """Merges intervals of the same dimensionality type.

    For example, two histograms with component histograms [10, 4, 0, 0] and
    [6, 2, 0, 0] are both 01D structures so they will be merged.

    Intervals are merged by summing the scores, and taking the minimum and
    maximum k-values.  Component IDs in the merged interval are taken from the
    interval with the highest score.

    On rare occasions, intervals to be merged are not adjacent.  In this case,
    the score of the merged interval is not equal to the score which would be
    calculated from its k-interval.  This is necessary to maintain the property
    that the scores sum to 1.
    """
    dimtypes = set([e.dimtype for e in intervals])
    merged_intervals = []
    for dimtype in dimtypes:
        relevant = [e for e in intervals if e.dimtype == dimtype]
        combined_score = sum([e.score for e in relevant])
        amin = min([e.a for e in relevant])
        bmax = max([e.b for e in relevant])
        best = max(relevant, key=lambda x: x.score)
        merged = build_kinterval(amin, bmax, best.h, best.components, best.cdim, score=combined_score)
        merged_intervals.append(merged)
    return merged_intervals