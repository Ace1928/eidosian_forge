import abc
import dataclasses
import functools
import inspect
import sys
from dataclasses import Field, fields
from typing import Any, Callable, Dict, Optional, Tuple, Union, Type, get_type_hints
from enum import Enum
from marshmallow.exceptions import ValidationError  # type: ignore
from dataclasses_json.utils import CatchAllVar
@staticmethod
def handle_from_dict(cls, kvs: Dict) -> Dict[str, Any]:
    known, unknown = _UndefinedParameterAction._separate_defined_undefined_kvs(cls=cls, kvs=kvs)
    catch_all_field = _CatchAllUndefinedParameters._get_catch_all_field(cls=cls)
    if catch_all_field.name in known:
        already_parsed = isinstance(known[catch_all_field.name], dict)
        default_value = _CatchAllUndefinedParameters._get_default(catch_all_field=catch_all_field)
        received_default = default_value == known[catch_all_field.name]
        value_to_write: Any
        if received_default and len(unknown) == 0:
            value_to_write = default_value
        elif received_default and len(unknown) > 0:
            value_to_write = unknown
        elif already_parsed:
            value_to_write = known[catch_all_field.name]
            if len(unknown) > 0:
                value_to_write.update(unknown)
        else:
            error_message = f"Received input field with same name as catch-all field: '{catch_all_field.name}': '{known[catch_all_field.name]}'"
            raise UndefinedParameterError(error_message)
    else:
        value_to_write = unknown
    known[catch_all_field.name] = value_to_write
    return known