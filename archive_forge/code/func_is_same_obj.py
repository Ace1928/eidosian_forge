import abc
import copy
from oslo_utils import reflection
from oslo_utils import strutils
from urllib import parse
from heatclient._i18n import _
from heatclient import exc as exceptions
def is_same_obj(self, other):
    """Identify the two objects are same one with same id."""
    if isinstance(other, self.__class__):
        if hasattr(self, 'id') and hasattr(other, 'id'):
            return self.id == other.id
    return False