from typing import Collection, List
from sys import maxsize
def is_printable_as_block_string(value: str) -> bool:
    """Check whether the given string is printable as a block string.

    For internal use only.
    """
    if not isinstance(value, str):
        value = str(value)
    if not value:
        return True
    is_empty_line = True
    has_indent = False
    has_common_indent = True
    seen_non_empty_line = False
    for c in value:
        if c == '\n':
            if is_empty_line and (not seen_non_empty_line):
                return False
            seen_non_empty_line = True
            is_empty_line = True
            has_indent = False
        elif c in ' \t':
            has_indent = has_indent or is_empty_line
        elif c <= '\x0f':
            return False
        else:
            has_common_indent = has_common_indent and has_indent
            is_empty_line = False
    if is_empty_line:
        return False
    if has_common_indent and seen_non_empty_line:
        return False
    return True