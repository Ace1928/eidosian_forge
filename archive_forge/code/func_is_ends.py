import abc
import copy
from neutron_lib import exceptions
@property
def is_ends(self):
    return bool(getattr(self, 'ends', False))