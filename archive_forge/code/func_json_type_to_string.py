import re
from ovs.db import error
def json_type_to_string(type_):
    number_types = [int]
    number_types.extend([float])
    number_types = tuple(number_types)
    if type_ is None:
        return 'null'
    elif issubclass(type_, bool):
        return 'boolean'
    elif issubclass(type_, dict):
        return 'object'
    elif issubclass(type_, list):
        return 'array'
    elif issubclass(type_, number_types):
        return 'number'
    elif issubclass(type_, str):
        return 'string'
    else:
        return '<invalid>'