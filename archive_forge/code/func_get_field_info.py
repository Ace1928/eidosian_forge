import json
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, ForwardRef, Optional, Tuple, Type, Union
from typing_extensions import Literal, Protocol
from .typing import AnyArgTCallable, AnyCallable
from .utils import GetterDict
from .version import compiled
@classmethod
def get_field_info(cls, name: str) -> Dict[str, Any]:
    """
        Get properties of FieldInfo from the `fields` property of the config class.
        """
    fields_value = cls.fields.get(name)
    if isinstance(fields_value, str):
        field_info: Dict[str, Any] = {'alias': fields_value}
    elif isinstance(fields_value, dict):
        field_info = fields_value
    else:
        field_info = {}
    if 'alias' in field_info:
        field_info.setdefault('alias_priority', 2)
    if field_info.get('alias_priority', 0) <= 1 and cls.alias_generator:
        alias = cls.alias_generator(name)
        if not isinstance(alias, str):
            raise TypeError(f'Config.alias_generator must return str, not {alias.__class__}')
        field_info.update(alias=alias, alias_priority=1)
    return field_info