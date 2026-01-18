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
@staticmethod
def next_field(field: FieldInfo | Any | None, key: str) -> FieldInfo | None:
    """
        Find the field in a sub model by key(env name)

        By having the following models:

            ```py
            class SubSubModel(BaseSettings):
                dvals: Dict

            class SubModel(BaseSettings):
                vals: list[str]
                sub_sub_model: SubSubModel

            class Cfg(BaseSettings):
                sub_model: SubModel
            ```

        Then:
            next_field(sub_model, 'vals') Returns the `vals` field of `SubModel` class
            next_field(sub_model, 'sub_sub_model') Returns `sub_sub_model` field of `SubModel` class

        Args:
            field: The field.
            key: The key (env name).

        Returns:
            Field if it finds the next field otherwise `None`.
        """
    if not field:
        return None
    annotation = field.annotation if isinstance(field, FieldInfo) else field
    if origin_is_union(get_origin(annotation)) or isinstance(annotation, WithArgsTypes):
        for type_ in get_args(annotation):
            type_has_key = EnvSettingsSource.next_field(type_, key)
            if type_has_key:
                return type_has_key
    elif is_model_class(annotation) and annotation.model_fields.get(key):
        return annotation.model_fields[key]
    return None