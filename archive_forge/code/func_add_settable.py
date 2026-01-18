from typing import (
from .constants import (
from .exceptions import (
from .utils import (
def add_settable(self, settable: Settable) -> None:
    """
        Convenience method to add a settable parameter to the CommandSet

        :param settable: Settable object being added
        """
    if self._cmd:
        if not self._cmd.always_prefix_settables:
            if settable.name in self._cmd.settables.keys() and settable.name not in self._settables.keys():
                raise KeyError(f'Duplicate settable: {settable.name}')
        else:
            prefixed_name = f'{self._settable_prefix}.{settable.name}'
            if prefixed_name in self._cmd.settables.keys() and settable.name not in self._settables.keys():
                raise KeyError(f'Duplicate settable: {settable.name}')
    self._settables[settable.name] = settable