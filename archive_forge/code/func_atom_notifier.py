import abc
from taskflow.types import notifier
from taskflow.utils import misc
@property
def atom_notifier(self):
    """The atom notifier."""
    return self._atom_notifier