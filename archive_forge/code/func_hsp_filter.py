from copy import deepcopy
from itertools import chain
from Bio.SearchIO._utils import optionalcascade
from ._base import _BaseSearchObject
from .hit import Hit
def hsp_filter(self, func=None):
    """Create new QueryResult object whose HSP objects pass the filter function.

        ``hsp_filter`` is the same as ``hit_filter``, except that it filters
        directly on each HSP object in every Hit. If the filtering removes
        all HSP objects in a given Hit, the entire Hit will be discarded. This
        will result in the QueryResult having less Hit after filtering.
        """
    hits = [x for x in (hit.filter(func) for hit in self.hits) if x]
    obj = self.__class__(hits, self.id, self._hit_key_function)
    self._transfer_attrs(obj)
    return obj