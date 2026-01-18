from collections import namedtuple
from math import floor, ceil
def draw_aalines(surf, color, closed, points, blend=True):
    """draw several anti-aliased lines connected through the points."""
    return _multi_lines(surf, color, closed, points, blend=blend, aaline=True)