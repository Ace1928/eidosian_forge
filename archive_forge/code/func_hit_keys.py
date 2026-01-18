from copy import deepcopy
from itertools import chain
from Bio.SearchIO._utils import optionalcascade
from ._base import _BaseSearchObject
from .hit import Hit
@property
def hit_keys(self):
    """Hit IDs of the Hit objects contained in the QueryResult."""
    return list(self._items.keys())