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
def is_pyarrow_version_less_than(v: str) -> bool:
    """Return True if the current Pyarrow version is less than the input version.

    Parameters
    ----------
    v : str
        Version string, e.g. "0.25.0"

    Returns
    -------
    bool

    """
    import pyarrow as pa
    from packaging import version
    return version.parse(pa.__version__) < version.parse(v)