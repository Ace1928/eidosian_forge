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
def is_bytes_like(obj: object) -> TypeGuard[BytesLike]:
    """True if the type is considered bytes-like for the purposes of
    protobuf data marshalling.
    """
    return isinstance(obj, _BYTES_LIKE_TYPES)