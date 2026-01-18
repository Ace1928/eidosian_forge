import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def is_some_sort_of_field_list(text: str, style: str) -> bool:
    """Determine if docstring contains field lists.

    Parameters
    ----------
    text : str
        The docstring text.
    style : str
        The field list style to use.

    Returns
    -------
    is_field_list : bool
        Whether the field list pattern for style was found in the docstring.
    """
    split_lines = text.rstrip().splitlines()
    if style == 'epytext':
        return any((re.match(EPYTEXT_REGEX, line) for line in split_lines))
    elif style == 'sphinx':
        return any((re.match(SPHINX_REGEX, line) for line in split_lines))
    return False