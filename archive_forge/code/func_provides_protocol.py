from heapq import heappop, heappush
import inspect
import itertools
import functools
from traits.adaptation.adaptation_error import AdaptationError
from traits.has_traits import HasTraits
from traits.trait_types import Dict, List, Str
@staticmethod
def provides_protocol(type_, protocol):
    """ Does the given type provide (i.e implement) a given protocol?

        Parameters
        ----------
        type_
            Python 'type'.
        protocol
            Either a regular Python class or a traits Interface.

        Returns
        -------
        result : bool
            True if the object provides the protocol, otherwise False.

        """
    return issubclass(type_, protocol)