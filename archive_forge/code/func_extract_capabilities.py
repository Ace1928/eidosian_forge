from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def extract_capabilities(text):
    """Extract a capabilities list from a string, if present.

    Args:
      text: String to extract from
    Returns: Tuple with text with capabilities removed and list of capabilities
    """
    if b'\x00' not in text:
        return (text, [])
    text, capabilities = text.rstrip().split(b'\x00')
    return (text, capabilities.strip().split(b' '))