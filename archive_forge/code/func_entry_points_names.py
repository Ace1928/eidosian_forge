import logging
import operator
from . import _cache
from .exception import NoMatches
def entry_points_names(self):
    """Return the list of entry points names for this namespace."""
    return list(map(operator.attrgetter('name'), self.list_entry_points()))