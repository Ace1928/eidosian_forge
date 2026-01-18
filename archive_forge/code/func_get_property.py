import warnings
from ..helpers import quote_string, random_string, stringify_param_value
from .commands import AsyncGraphCommands, GraphCommands
from .edge import Edge  # noqa
from .node import Node  # noqa
from .path import Path  # noqa
def get_property(self, idx):
    """
        Returns a property by it's index

        Args:

        idx:
            The index of the property
        """
    try:
        p = self._properties[idx]
    except IndexError:
        self._refresh_attributes()
        p = self._properties[idx]
    return p