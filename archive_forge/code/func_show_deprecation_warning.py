from __future__ import annotations
import functools
from typing import Any, Callable, Final, TypeVar, cast
import streamlit
from streamlit import config
from streamlit.logger import get_logger
def show_deprecation_warning(message: str) -> None:
    """Show a deprecation warning message."""
    if _should_show_deprecation_warning_in_browser():
        streamlit.warning(message)
    _LOGGER.warning(message)