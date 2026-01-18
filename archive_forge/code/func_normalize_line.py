import contextlib
import re
from typing import List, Match, Optional, Union
def normalize_line(line: str, newline: str) -> str:
    """Return line with fixed ending, if ending was present in line.

    Otherwise, does nothing.

    Parameters
    ----------
    line : str
        The line to normalize.
    newline : str
        The newline character to use for line endings.

    Returns
    -------
    normalized_line : str
        The supplied line with line endings replaced by the newline.
    """
    stripped = line.rstrip('\n\r')
    return stripped + newline if stripped != line else line