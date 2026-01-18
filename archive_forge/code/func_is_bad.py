import warnings
import re
def is_bad(text: str) -> bool:
    """
    Returns true iff the given text looks like it contains mojibake.

    This can be faster than `badness`, because it returns when the first match
    is found to a regex instead of counting matches. Note that as strings get
    longer, they have a higher chance of returning True for `is_bad(string)`.
    """
    return bool(BADNESS_RE.search(text))