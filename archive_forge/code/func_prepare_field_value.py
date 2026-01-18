from __future__ import annotations as _annotations
import json
import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Sequence, Tuple, Union, cast
from dotenv import dotenv_values
from pydantic import AliasChoices, AliasPath, BaseModel, Json
from pydantic._internal._typing_extra import WithArgsTypes, origin_is_union
from pydantic._internal._utils import deep_update, is_model_class, lenient_issubclass
from pydantic.fields import FieldInfo
from typing_extensions import get_args, get_origin
from pydantic_settings.utils import path_type_label
def prepare_field_value(self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool) -> Any:
    """
        Prepare value for the field.

        * Extract value for nested field.
        * Deserialize value to python object for complex field.

        Args:
            field: The field.
            field_name: The field name.

        Returns:
            A tuple contains prepared value for the field.

        Raises:
            ValuesError: When There is an error in deserializing value for complex field.
        """
    is_complex, allow_parse_failure = self._field_is_complex(field)
    if is_complex or value_is_complex:
        if value is None:
            env_val_built = self.explode_env_vars(field_name, field, self.env_vars)
            if env_val_built:
                return env_val_built
        else:
            try:
                value = self.decode_complex_value(field_name, field, value)
            except ValueError as e:
                if not allow_parse_failure:
                    raise e
            if isinstance(value, dict):
                return deep_update(value, self.explode_env_vars(field_name, field, self.env_vars))
            else:
                return value
    elif value is not None:
        return value