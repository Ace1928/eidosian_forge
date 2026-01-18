import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def is_some_sort_of_code(text: str) -> bool:
    """Return True if text looks like code."""
    return any((len(word) > 50 and (not re.match(URL_REGEX, word)) for word in text.split()))