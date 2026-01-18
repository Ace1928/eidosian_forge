from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.utils import get_bolded_text
from langchain_core.utils._merge import merge_dicts
from langchain_core.utils.interactive_env import is_interactive_env
def get_msg_title_repr(title: str, *, bold: bool=False) -> str:
    """Get a title representation for a message.

    Args:
        title: The title.
        bold: Whether to bold the title.

    Returns:
        The title representation.
    """
    padded = ' ' + title + ' '
    sep_len = (80 - len(padded)) // 2
    sep = '=' * sep_len
    second_sep = sep + '=' if len(padded) % 2 else sep
    if bold:
        padded = get_bolded_text(padded)
    return f'{sep}{padded}{second_sep}'