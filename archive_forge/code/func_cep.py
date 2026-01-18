import random
from .links_base import CrossingEntryPoint
from ..sage_helper import _within_sage
def cep(crossing_strand):
    """
    Returns the CrossingEntryPoint corresponding to the given
    CrossingStrand in the same crossing; that is, it orients the
    CrossingStrand without changing the crossing.
    """
    if crossing_strand == crossing_strand.oriented():
        return crossing_strand
    else:
        return crossing_strand.rotate(2)