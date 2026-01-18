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
def tagged_union_schema(self, schema: core_schema.TaggedUnionSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches a schema that allows values matching any of the given schemas, where
        the schemas are tagged with a discriminator field that indicates which schema should be used to validate
        the value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    generated: dict[str, JsonSchemaValue] = {}
    for k, v in schema['choices'].items():
        if isinstance(k, Enum):
            k = k.value
        try:
            generated[str(k)] = self.generate_inner(v).copy()
        except PydanticOmit:
            continue
        except PydanticInvalidForJsonSchema as exc:
            self.emit_warning('skipped-choice', exc.message)
    one_of_choices = _deduplicate_schemas(generated.values())
    json_schema: JsonSchemaValue = {'oneOf': one_of_choices}
    openapi_discriminator = self._extract_discriminator(schema, one_of_choices)
    if openapi_discriminator is not None:
        json_schema['discriminator'] = {'propertyName': openapi_discriminator, 'mapping': {k: v.get('$ref', v) for k, v in generated.items()}}
    return json_schema