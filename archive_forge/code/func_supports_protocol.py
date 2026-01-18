from heapq import heappop, heappush
import inspect
import itertools
import functools
from traits.adaptation.adaptation_error import AdaptationError
from traits.has_traits import HasTraits
from traits.trait_types import Dict, List, Str
def supports_protocol(self, obj, protocol):
    """ Does the object support a given protocol?

        An object "supports" a protocol if either it "provides" it directly,
        or it can be adapted to it.

        """
    return self.adapt(obj, protocol, None) is not None