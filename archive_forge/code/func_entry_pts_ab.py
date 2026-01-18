import random
from .links_base import CrossingEntryPoint
from ..sage_helper import _within_sage
def entry_pts_ab(crossing):
    """
    The two entry points of a crossing with Dror's convention that the
    overcrossing ("a") is first and the undercrossing ("b") is second.
    """
    verts = [1, 0] if crossing.sign == -1 else [3, 0]
    return [CrossingEntryPoint(crossing, v) for v in verts]