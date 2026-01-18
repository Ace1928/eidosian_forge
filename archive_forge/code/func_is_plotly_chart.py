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
def is_plotly_chart(obj: object) -> TypeGuard[Figure | list[Any] | dict[str, Any]]:
    """True if input looks like a Plotly chart."""
    return is_type(obj, 'plotly.graph_objs._figure.Figure') or _is_list_of_plotly_objs(obj) or _is_probably_plotly_dict(obj)