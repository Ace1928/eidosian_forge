from .links import CrossingStrand
from ..graphs import CyclicList
def is_end_of_twist_region(crossing) -> bool:
    return len(set((C for C, i in crossing.adjacent))) == 3