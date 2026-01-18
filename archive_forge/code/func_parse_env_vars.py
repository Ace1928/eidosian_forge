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
def parse_env_vars(env_vars: Mapping[str, str | None], case_sensitive: bool=False, ignore_empty: bool=False, parse_none_str: str | None=None) -> Mapping[str, str | None]:
    return {_get_env_var_key(k, case_sensitive): _parse_env_none_str(v, parse_none_str) for k, v in env_vars.items() if not (ignore_empty and v == '')}