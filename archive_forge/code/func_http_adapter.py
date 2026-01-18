import inspect
from typing import Any, Callable, Optional, Type, Union
from fastapi import Body
from ray._private.pydantic_compat import BaseModel
from ray._private.utils import import_attr
from ray.util.annotations import DeveloperAPI
def http_adapter(inp: http_adapter=Body(...)):
    return inp