from functools import partial
from inspect import isclass
from ..types import Field, Interface, ObjectType
from ..types.interface import InterfaceOptions
from ..types.utils import get_type
from .id_type import BaseGlobalIDType, DefaultGlobalIDType
@classmethod
def node_resolver(cls, only_type, root, info, id):
    return cls.get_node_from_global_id(info, id, only_type=only_type)