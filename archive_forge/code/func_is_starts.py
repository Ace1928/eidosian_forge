import abc
import copy
from neutron_lib import exceptions
@property
def is_starts(self):
    return bool(getattr(self, 'starts', False))