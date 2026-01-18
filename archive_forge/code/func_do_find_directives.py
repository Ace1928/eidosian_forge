import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def do_find_directives(text: str) -> bool:
    """Determine if docstring contains any reST directives.

    .. todo::

        Currently this function only returns True/False to indicate whether a
        reST directive was found.  Should return a list of tuples containing
        the start and end position of each reST directive found similar to the
        function do_find_links().

    Parameters
    ----------
    text : str
        The docstring text to test.

    Returns
    -------
    is_directive : bool
        Whether the docstring is a reST directive.
    """
    _rest_iter = re.finditer(REST_REGEX, text)
    return bool([(_rest.start(0), _rest.end(0)) for _rest in _rest_iter])