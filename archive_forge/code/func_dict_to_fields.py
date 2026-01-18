from typing import Optional, Union
from .helpers import LineKey, PCColumn
from .util import Attr, Panel, coalesce, nested_get, nested_set
from .validators import (
def dict_to_fields(d):
    fields = []
    for k, v in d.items():
        if isinstance(v, dict) and len(v) > 0:
            field = {'name': k, 'args': dict_to_fields(v), 'fields': []}
        elif isinstance(v, dict) and len(v) == 0 or v is None:
            field = {'name': k, 'fields': []}
        else:
            field = {'name': k, 'value': v}
        fields.append(field)
    return fields