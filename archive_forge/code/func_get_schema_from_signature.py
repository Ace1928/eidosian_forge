import inspect
import json
import re
from typing import Callable, Optional
from jsonschema.protocols import Validator
from pydantic import create_model
from referencing import Registry, Resource
from referencing._core import Resolver
from referencing.jsonschema import DRAFT202012
def get_schema_from_signature(fn: Callable) -> str:
    """Turn a function signature into a JSON schema.

    Every JSON object valid to the output JSON Schema can be passed
    to `fn` using the ** unpacking syntax.

    """
    signature = inspect.signature(fn)
    arguments = {}
    for name, arg in signature.parameters.items():
        if arg.annotation == inspect._empty:
            raise ValueError('Each argument must have a type annotation')
        else:
            arguments[name] = (arg.annotation, ...)
    model = create_model('Arguments', **arguments)
    return model.model_json_schema()