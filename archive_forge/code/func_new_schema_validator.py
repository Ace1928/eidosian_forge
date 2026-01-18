from __future__ import annotations
from typing import Any, Callable, NamedTuple
from pydantic_core import CoreConfig, CoreSchema, ValidationError
from typing_extensions import Literal, Protocol, TypeAlias
def new_schema_validator(self, schema: CoreSchema, schema_type: Any, schema_type_path: SchemaTypePath, schema_kind: SchemaKind, config: CoreConfig | None, plugin_settings: dict[str, object]) -> tuple[ValidatePythonHandlerProtocol | None, ValidateJsonHandlerProtocol | None, ValidateStringsHandlerProtocol | None]:
    """This method is called for each plugin every time a new [`SchemaValidator`][pydantic_core.SchemaValidator]
        is created.

        It should return an event handler for each of the three validation methods, or `None` if the plugin does not
        implement that method.

        Args:
            schema: The schema to validate against.
            schema_type: The original type which the schema was created from, e.g. the model class.
            schema_type_path: Path defining where `schema_type` was defined, or where `TypeAdapter` was called.
            schema_kind: The kind of schema to validate against.
            config: The config to use for validation.
            plugin_settings: Any plugin settings.

        Returns:
            A tuple of optional event handlers for each of the three validation methods -
                `validate_python`, `validate_json`, `validate_strings`.
        """
    raise NotImplementedError('Pydantic plugins should implement `new_schema_validator`.')