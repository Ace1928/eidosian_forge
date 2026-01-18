from fontTools.ttLib.ttGlyphSet import LerpGlyphSet
from fontTools.pens.basePen import AbstractPen, BasePen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, SegmentToPointPen
from fontTools.pens.recordingPen import RecordingPen, DecomposingRecordingPen
from fontTools.misc.transform import Transform
from collections import defaultdict, deque
from math import sqrt, copysign, atan2, pi
from enum import Enum
import itertools
import logging
def matching_for_vectors(m0, m1):
    n = len(m0)
    identity_matching = list(range(n))
    costs = [[vdiff_hypot2(v0, v1) for v1 in m1] for v0 in m0]
    matching, matching_cost = min_cost_perfect_bipartite_matching(costs)
    identity_cost = sum((costs[i][i] for i in range(n)))
    return (matching, matching_cost, identity_cost)