from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.utils import get_bolded_text
from langchain_core.utils._merge import merge_dicts
from langchain_core.utils.interactive_env import is_interactive_env
def merge_content(first_content: Union[str, List[Union[str, Dict]]], second_content: Union[str, List[Union[str, Dict]]]) -> Union[str, List[Union[str, Dict]]]:
    """Merge two message contents.

    Args:
        first_content: The first content.
        second_content: The second content.

    Returns:
        The merged content.
    """
    if isinstance(first_content, str):
        if isinstance(second_content, str):
            return first_content + second_content
        else:
            return_list: List[Union[str, Dict]] = [first_content]
            return return_list + second_content
    elif isinstance(second_content, List):
        return first_content + second_content
    elif isinstance(first_content[-1], str):
        return first_content[:-1] + [first_content[-1] + second_content]
    else:
        return first_content + [second_content]