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
def generate_definitions(self, inputs: Sequence[tuple[JsonSchemaKeyT, JsonSchemaMode, core_schema.CoreSchema]]) -> tuple[dict[tuple[JsonSchemaKeyT, JsonSchemaMode], JsonSchemaValue], dict[DefsRef, JsonSchemaValue]]:
    """Generates JSON schema definitions from a list of core schemas, pairing the generated definitions with a
        mapping that links the input keys to the definition references.

        Args:
            inputs: A sequence of tuples, where:

                - The first element is a JSON schema key type.
                - The second element is the JSON mode: either 'validation' or 'serialization'.
                - The third element is a core schema.

        Returns:
            A tuple where:

                - The first element is a dictionary whose keys are tuples of JSON schema key type and JSON mode, and
                    whose values are the JSON schema corresponding to that pair of inputs. (These schemas may have
                    JsonRef references to definitions that are defined in the second returned element.)
                - The second element is a dictionary whose keys are definition references for the JSON schemas
                    from the first returned element, and whose values are the actual JSON schema definitions.

        Raises:
            PydanticUserError: Raised if the JSON schema generator has already been used to generate a JSON schema.
        """
    if self._used:
        raise PydanticUserError(f'This JSON schema generator has already been used to generate a JSON schema. You must create a new instance of {type(self).__name__} to generate a new JSON schema.', code='json-schema-already-used')
    for key, mode, schema in inputs:
        self._mode = mode
        self.generate_inner(schema)
    definitions_remapping = self._build_definitions_remapping()
    json_schemas_map: dict[tuple[JsonSchemaKeyT, JsonSchemaMode], DefsRef] = {}
    for key, mode, schema in inputs:
        self._mode = mode
        json_schema = self.generate_inner(schema)
        json_schemas_map[key, mode] = definitions_remapping.remap_json_schema(json_schema)
    json_schema = {'$defs': self.definitions}
    json_schema = definitions_remapping.remap_json_schema(json_schema)
    self._used = True
    return (json_schemas_map, _sort_json_schema(json_schema['$defs']))