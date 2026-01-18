from heapq import heappop, heappush
import inspect
import itertools
import functools
from traits.adaptation.adaptation_error import AdaptationError
from traits.has_traits import HasTraits
from traits.trait_types import Dict, List, Str
def get_global_adaptation_manager():
    """ Set a reference to the global adaptation manager. """
    global adaptation_manager
    return adaptation_manager