from __future__ import annotations
import re
from typing import Any
def get_method_args_from_code(args: list[Any], line: str) -> list[str]:
    """Parse arguments from a stringified arguments list inside parentheses

    Parameters
    ----------
    args : list
        A list where it's size matches the expected number of parsed arguments
    line : str
        Stringified line of code with method arguments inside parentheses

    Returns
    -------
    list of strings
        Parsed arguments

    Example
    -------
    >>> line = 'foo(bar, baz, my(func, tion))'
    >>>
    >>> get_method_args_from_code(range(0, 3), line)
    ['bar', 'baz', 'my(func, tion)']

    """
    line_args = extract_args(line)[0]
    if len(args) > 1:
        inputs = re.split(',\\s*(?![^(){}[\\]]*\\))', line_args)
        assert len(inputs) == len(args), 'Could not split arguments'
    else:
        inputs = [line_args]
    return inputs