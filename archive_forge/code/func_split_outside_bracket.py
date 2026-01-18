import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union
def split_outside_bracket(line: str, delimiter: str=',') -> List[str]:
    """Given a line of text, split it on comma unless the comma is within a bracket '[]'."""
    bracket_count = 0
    curr_token = ''
    res = []
    for char in line:
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
        elif char == delimiter and bracket_count == 0:
            res.append(curr_token)
            curr_token = ''
            continue
        curr_token += char
    res.append(curr_token)
    return res