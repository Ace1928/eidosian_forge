import warnings
from ..helpers import quote_string, random_string, stringify_param_value
from .commands import AsyncGraphCommands, GraphCommands
from .edge import Edge  # noqa
from .node import Node  # noqa
from .path import Path  # noqa
def property_keys(self):
    return self.call_procedure(DB_PROPERTYKEYS, read_only=True).result_set