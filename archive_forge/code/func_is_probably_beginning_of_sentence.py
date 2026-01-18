import contextlib
import re
from typing import List, Match, Optional, Union
def is_probably_beginning_of_sentence(line: str) -> Union[Match[str], None, bool]:
    """Determine if the line begins a sentence.

    Parameters
    ----------
    line : str
        The line to be tested.

    Returns
    -------
    is_beginning : bool
        True if this token is the beginning of a sentence.
    """
    for token in ['@', '-', '\\*']:
        if re.search(f'\\s{token}\\s', line):
            return True
    stripped_line = line.strip()
    is_beginning_of_sentence = re.match('^[-@\\)]', stripped_line)
    is_pydoc_ref = re.match('^:\\w+:', stripped_line)
    return is_beginning_of_sentence and (not is_pydoc_ref)