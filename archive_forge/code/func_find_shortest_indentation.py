import contextlib
import re
from typing import List, Match, Optional, Union
def find_shortest_indentation(lines: List[str]) -> str:
    """Determine the shortest indentation in a list of lines.

    Parameters
    ----------
    lines : list
        A list of lines to check indentation.

    Returns
    -------
    indentation : str
        The shortest (smallest number of spaces) indentation in the list of
        lines.
    """
    assert not isinstance(lines, str)
    indentation = None
    for line in lines:
        if line.strip():
            non_whitespace_index = len(line) - len(line.lstrip())
            _indent = line[:non_whitespace_index]
            if indentation is None or len(_indent) < len(indentation):
                indentation = _indent
    return indentation or ''