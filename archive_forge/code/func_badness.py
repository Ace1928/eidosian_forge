import warnings
import re
def badness(text: str) -> int:
    """
    Get the 'badness' of a sequence of text, counting the number of unlikely
    character sequences. A badness greater than 0 indicates that some of it
    seems to be mojibake.
    """
    return len(BADNESS_RE.findall(text))