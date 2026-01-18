from __future__ import annotations
from collections import defaultdict
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable
from pydantic_core import CoreSchema, PydanticCustomError, to_jsonable_python
from pydantic_core import core_schema as cs
from ._fields import PydanticMetadata
def add_js_update_schema(s: cs.CoreSchema, f: Callable[[], dict[str, Any]]) -> None:

    def update_js_schema(s: cs.CoreSchema, handler: GetJsonSchemaHandler) -> dict[str, Any]:
        js_schema = handler(s)
        js_schema.update(f())
        return js_schema
    if 'metadata' in s:
        metadata = s['metadata']
        if 'pydantic_js_functions' in s:
            metadata['pydantic_js_functions'].append(update_js_schema)
        else:
            metadata['pydantic_js_functions'] = [update_js_schema]
    else:
        s['metadata'] = {'pydantic_js_functions': [update_js_schema]}