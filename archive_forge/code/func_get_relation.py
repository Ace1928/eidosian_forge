import warnings
from ..helpers import quote_string, random_string, stringify_param_value
from .commands import AsyncGraphCommands, GraphCommands
from .edge import Edge  # noqa
from .node import Node  # noqa
from .path import Path  # noqa
def get_relation(self, idx):
    """
        Returns a relationship type by it's index

        Args:

        idx:
            The index of the relation
        """
    try:
        relationship_type = self._relationship_types[idx]
    except IndexError:
        self._refresh_relations()
        relationship_type = self._relationship_types[idx]
    return relationship_type