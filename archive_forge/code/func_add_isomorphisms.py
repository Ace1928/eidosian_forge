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
def add_isomorphisms(points, isomorphisms, reverse):
    reference_bits = points_characteristic_bits(points)
    n = len(points)
    if reverse:
        points = points[::-1]
        bits = points_characteristic_bits(points)
    else:
        bits = reference_bits
    vector = points_complex_vector(points)
    assert len(vector) % n == 0
    mult = len(vector) // n
    mask = (1 << n) - 1
    for i in range(n):
        b = bits << n - i & mask | bits >> i
        if b == reference_bits:
            isomorphisms.append((rot_list(vector, -i * mult), n - 1 - i if reverse else i, reverse))