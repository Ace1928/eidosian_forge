import difflib
import os
import sys
import textwrap
from typing import Any, Optional, Tuple, Union
def locale_escape(string: Any, errors: str='replace') -> str:
    """Mangle non-supported characters, for savages with ASCII terminals.

    string (Any): The string to escape.
    errors (str): The str.encode errors setting. Defaults to `"replace"`.
    RETURNS (str): The escaped string.
    """
    string = str(string)
    string = string.encode(ENCODING, errors).decode('utf8')
    return string