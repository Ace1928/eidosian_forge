import contextlib
import re
from typing import List, Match, Optional, Union
def normalize_summary(summary: str, noncap: Optional[List[str]]=None) -> str:
    """Return normalized docstring summary.

    A normalized docstring summary will have the first word capitalized and
    a period at the end.

    Parameters
    ----------
    summary : str
        The summary string.
    noncap : list
        A user-provided list of words not to capitalize when they appear as
        the first word in the summary.

    Returns
    -------
    summary : str
        The normalized summary string.
    """
    if noncap is None:
        noncap = []
    summary = summary.rstrip()
    if summary and (summary[-1].isalnum() or summary[-1] in ['"', "'"]) and (not summary.startswith('#')):
        summary += '.'
    with contextlib.suppress(IndexError):
        if all((char not in summary.split(' ', 1)[0] for char in ['_', '.'])) and summary.split(' ', 1)[0] not in noncap:
            summary = summary[0].upper() + summary[1:]
    return summary