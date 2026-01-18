from functools import partial
from inspect import isclass
from ..types import Field, Interface, ObjectType
from ..types.interface import InterfaceOptions
from ..types.utils import get_type
from .id_type import BaseGlobalIDType, DefaultGlobalIDType
@staticmethod
def id_resolver(parent_resolver, node, root, info, parent_type_name=None, **args):
    type_id = parent_resolver(root, info, **args)
    parent_type_name = parent_type_name or info.parent_type.name
    return node.to_global_id(parent_type_name, type_id)