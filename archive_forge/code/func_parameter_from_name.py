import operator
import typing
from collections.abc import MappingView, MutableMapping, MutableSet
def parameter_from_name(self, name: str, default: typing.Any=None):
    """Get a :class:`.Parameter` with references in this table by its string name.

        If the parameter is not present, return the ``default`` value.

        Args:
            name: The name of the :class:`.Parameter`
            default: The object that should be returned if the parameter is missing.
        """
    return self._names.get(name, default)