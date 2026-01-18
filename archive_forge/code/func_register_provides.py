from heapq import heappop, heappush
import inspect
import itertools
import functools
from traits.adaptation.adaptation_error import AdaptationError
from traits.has_traits import HasTraits
from traits.trait_types import Dict, List, Str
def register_provides(self, provider_protocol, protocol):
    """ Register that a protocol provides another. """
    self.register_factory(no_adapter_necessary, provider_protocol, protocol)