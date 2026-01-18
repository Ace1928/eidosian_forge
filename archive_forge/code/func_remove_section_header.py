import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def remove_section_header(text):
    """Return text with section header removed.

    >>> remove_section_header('----\\nfoo\\nbar\\n')
    'foo\\nbar\\n'

    >>> remove_section_header('===\\nfoo\\nbar\\n')
    'foo\\nbar\\n'
    """
    stripped = text.lstrip()
    if not stripped:
        return text
    first = stripped[0]
    return text if first.isalnum() or first.isspace() or stripped.splitlines()[0].strip(first).strip() else stripped.lstrip(first).lstrip()