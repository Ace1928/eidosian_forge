from typing import Optional, Union
from .helpers import LineKey, PCColumn
from .util import Attr, Panel, coalesce, nested_get, nested_set
from .validators import (
def fields_to_dict(fields):
    d = {}
    for field in fields:
        keys = set(field.keys())
        name = field['name']
        if keys == {'name', 'fields'}:
            d[name] = {}
        elif keys == {'name', 'value'}:
            d[name] = field['value']
        elif keys == {'name', 'args', 'fields'}:
            d[name] = fields_to_dict(field['args'])
    return d