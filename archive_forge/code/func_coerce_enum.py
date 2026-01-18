from __future__ import annotations
import contextlib
import copy
import math
import re
import types
from enum import Enum, EnumMeta, auto
from typing import (
from typing_extensions import TypeAlias, TypeGuard
import streamlit as st
from streamlit import config, errors
from streamlit import logger as _logger
from streamlit import string_util
from streamlit.errors import StreamlitAPIException
def coerce_enum(from_enum_value: E1, to_enum_class: type[E2]) -> E1 | E2:
    """Attempt to coerce an Enum value to another EnumMeta.

    An Enum value of EnumMeta E1 is considered coercable to EnumType E2
    if the EnumMeta __qualname__ match and the names of their members
    match as well. (This is configurable in streamlist configs)
    """
    if not isinstance(from_enum_value, Enum):
        raise ValueError(f'Expected an Enum in the first argument. Got {type(from_enum_value)}')
    if not isinstance(to_enum_class, EnumMeta):
        raise ValueError(f'Expected an EnumMeta/Type in the second argument. Got {type(to_enum_class)}')
    if isinstance(from_enum_value, to_enum_class):
        return from_enum_value
    coercion_type = config.get_option('runner.enumCoercion')
    if coercion_type not in ALLOWED_ENUM_COERCION_CONFIG_SETTINGS:
        raise errors.StreamlitAPIException(f"Invalid value for config option runner.enumCoercion. Expected one of {ALLOWED_ENUM_COERCION_CONFIG_SETTINGS}, but got '{coercion_type}'.")
    if coercion_type == 'off':
        return from_enum_value
    from_enum_class = from_enum_value.__class__
    if from_enum_class.__qualname__ != to_enum_class.__qualname__ or (coercion_type == 'nameOnly' and set(to_enum_class._member_names_) != set(from_enum_class._member_names_)) or (coercion_type == 'nameAndValue' and set(to_enum_class._value2member_map_) != set(from_enum_class._value2member_map_)):
        _LOGGER.debug('Failed to coerce %s to class %s', from_enum_value, to_enum_class)
        return from_enum_value
    _LOGGER.debug('Coerced %s to class %s', from_enum_value, to_enum_class)
    return to_enum_class[from_enum_value._name_]