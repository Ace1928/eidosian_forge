from __future__ import annotations
import functools
from typing import Any, Callable, Final, TypeVar, cast
import streamlit
from streamlit import config
from streamlit.logger import get_logger
def make_deprecated_name_warning(old_name: str, new_name: str, removal_date: str, extra_message: str | None=None, include_st_prefix: bool=True) -> str:
    if include_st_prefix:
        old_name = f'st.{old_name}'
        new_name = f'st.{new_name}'
    return f'Please replace `{old_name}` with `{new_name}`.\n\n`{old_name}` will be removed after {removal_date}.' + (f'\n\n{extra_message}' if extra_message else '')