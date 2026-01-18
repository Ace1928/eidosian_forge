import random
from .links_base import CrossingEntryPoint
from ..sage_helper import _within_sage
def num_overlap(crossing, frontier):
    neighbor_strands = set((cs.opposite() for cs in crossing.crossing_strands()))
    return len(neighbor_strands.intersection(frontier))