from copy import deepcopy
from itertools import chain
from Bio.SearchIO._utils import optionalcascade
from ._base import _BaseSearchObject
from .hit import Hit
def hsp_map(self, func=None):
    """Create new QueryResult object, mapping the given function to its HSPs.

        ``hsp_map`` is the same as ``hit_map``, except that it applies the given
        function to all HSP objects in every Hit, instead of the Hit objects.
        """
    hits = [x for x in (hit.map(func) for hit in list(self.hits)[:]) if x]
    obj = self.__class__(hits, self.id, self._hit_key_function)
    self._transfer_attrs(obj)
    return obj