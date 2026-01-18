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
def maybe_tuple_to_list(item: Any) -> Any:
    """Convert a tuple to a list. Leave as is if it's not a tuple."""
    return list(item) if isinstance(item, tuple) else item