from __future__ import annotations as _annotations
import dataclasses
import inspect
import math
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import is_dataclass
from enum import Enum
from typing import (
import pydantic_core
from pydantic_core import CoreSchema, PydanticOmit, core_schema, to_jsonable_python
from pydantic_core.core_schema import ComputedField
from typing_extensions import Annotated, Literal, TypeAlias, assert_never, deprecated, final
from pydantic.warnings import PydanticDeprecatedSince26
from ._internal import (
from .annotated_handlers import GetJsonSchemaHandler
from .config import JsonDict, JsonSchemaExtraCallable, JsonValue
from .errors import PydanticInvalidForJsonSchema, PydanticUserError
def render_warning_message(self, kind: JsonSchemaWarningKind, detail: str) -> str | None:
    """This method is responsible for ignoring warnings as desired, and for formatting the warning messages.

        You can override the value of `ignored_warning_kinds` in a subclass of GenerateJsonSchema
        to modify what warnings are generated. If you want more control, you can override this method;
        just return None in situations where you don't want warnings to be emitted.

        Args:
            kind: The kind of warning to render. It can be one of the following:

                - 'skipped-choice': A choice field was skipped because it had no valid choices.
                - 'non-serializable-default': A default value was skipped because it was not JSON-serializable.
            detail: A string with additional details about the warning.

        Returns:
            The formatted warning message, or `None` if no warning should be emitted.
        """
    if kind in self.ignored_warning_kinds:
        return None
    return f'{detail} [{kind}]'