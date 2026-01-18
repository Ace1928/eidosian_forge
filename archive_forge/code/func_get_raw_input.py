import difflib
import os
import sys
import textwrap
from typing import Any, Optional, Tuple, Union
def get_raw_input(description: str, default: Optional[Union[str, bool]]=False, indent: int=4) -> str:
    """Get user input from the command line via raw_input / input.

    description (str): Text to display before prompt.
    default (Optional[Union[str, bool]]): Optional default value to display with prompt.
    indent (int): Indentation in spaces.
    RETURNS (str): User input.
    """
    additional = ' (default: {})'.format(default) if default else ''
    prompt = wrap('{}{}: '.format(description, additional), indent=indent)
    user_input = input(prompt)
    return user_input