from __future__ import annotations
import functools
from typing import Any, Callable, Final, TypeVar, cast
import streamlit
from streamlit import config
from streamlit.logger import get_logger
def maybe_show_warning() -> None:
    nonlocal has_shown_warning
    if not has_shown_warning:
        has_shown_warning = True
        show_warning()