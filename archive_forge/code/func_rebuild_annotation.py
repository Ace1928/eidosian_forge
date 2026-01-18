from __future__ import annotations as _annotations
import dataclasses
import inspect
import typing
from copy import copy
from dataclasses import Field as DataclassField
from functools import cached_property
from typing import Any, ClassVar
from warnings import warn
import annotated_types
import typing_extensions
from pydantic_core import PydanticUndefined
from typing_extensions import Literal, Unpack
from . import types
from ._internal import _decorators, _fields, _generics, _internal_dataclass, _repr, _typing_extra, _utils
from .aliases import AliasChoices, AliasPath
from .config import JsonDict
from .errors import PydanticUserError
from .warnings import PydanticDeprecatedSince20
def rebuild_annotation(self) -> Any:
    """Attempts to rebuild the original annotation for use in function signatures.

        If metadata is present, it adds it to the original annotation using
        `Annotated`. Otherwise, it returns the original annotation as-is.

        Note that because the metadata has been flattened, the original annotation
        may not be reconstructed exactly as originally provided, e.g. if the original
        type had unrecognized annotations, or was annotated with a call to `pydantic.Field`.

        Returns:
            The rebuilt annotation.
        """
    if not self.metadata:
        return self.annotation
    else:
        return typing_extensions.Annotated[self.annotation, *self.metadata]