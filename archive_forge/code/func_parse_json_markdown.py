from __future__ import annotations
import json
import re
from typing import Any, Callable, List
from langchain_core.exceptions import OutputParserException
def parse_json_markdown(json_string: str, *, parser: Callable[[str], Any]=parse_partial_json) -> dict:
    """
    Parse a JSON string from a Markdown string.

    Args:
        json_string: The Markdown string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    try:
        return _parse_json(json_string, parser=parser)
    except json.JSONDecodeError:
        match = re.search('```(json)?(.*)', json_string, re.DOTALL)
        if match is None:
            json_str = json_string
        else:
            json_str = match.group(2)
    return _parse_json(json_str, parser=parser)