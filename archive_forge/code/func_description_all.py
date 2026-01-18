from itertools import chain
from Bio.SearchIO._utils import allitems, optionalcascade, getattr_str
from ._base import _BaseSearchObject
from .hsp import HSP
@property
def description_all(self):
    """Alternative descriptions of the Hit."""
    return [self.description] + self._description_alt