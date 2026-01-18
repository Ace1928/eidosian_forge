import re
from typing import Optional, Tuple, cast
from .ast import Location
from .location import SourceLocation, get_location
from .source import Source
def print_prefixed_lines(*lines: Tuple[str, Optional[str]]) -> str:
    """Print lines specified like this: ("prefix", "string")"""
    existing_lines = [cast(Tuple[str, str], line) for line in lines if line[1] is not None]
    pad_len = max((len(line[0]) for line in existing_lines))
    return '\n'.join((prefix.rjust(pad_len) + (' ' + line if line else '') for prefix, line in existing_lines))