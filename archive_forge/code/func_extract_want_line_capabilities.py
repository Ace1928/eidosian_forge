from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def extract_want_line_capabilities(text):
    """Extract a capabilities list from a want line, if present.

    Note that want lines have capabilities separated from the rest of the line
    by a space instead of a null byte. Thus want lines have the form:

        want obj-id cap1 cap2 ...

    Args:
      text: Want line to extract from
    Returns: Tuple with text with capabilities removed and list of capabilities
    """
    split_text = text.rstrip().split(b' ')
    if len(split_text) < 3:
        return (text, [])
    return (b' '.join(split_text[:2]), split_text[2:])