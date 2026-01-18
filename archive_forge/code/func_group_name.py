from abc import ABCMeta
from abc import abstractmethod
from ray.util.collective.types import (
@property
def group_name(self):
    """Return the group name of this group."""
    return self._group_name