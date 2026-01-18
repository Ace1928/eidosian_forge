from __future__ import annotations as _annotations
import warnings
from contextlib import contextmanager
from typing import (
from pydantic_core import core_schema
from typing_extensions import (
from ..aliases import AliasGenerator
from ..config import ConfigDict, ExtraValues, JsonDict, JsonEncoder, JsonSchemaExtraCallable
from ..errors import PydanticUserError
from ..warnings import PydanticDeprecatedSince20
def dict_not_none(**kwargs: Any) -> Any:
    return {k: v for k, v in kwargs.items() if v is not None}