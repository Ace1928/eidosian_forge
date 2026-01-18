import email.parser
import time
from difflib import SequenceMatcher
from typing import BinaryIO, Optional, TextIO, Union
from .objects import S_ISGITLINK, Blob, Commit
from .pack import ObjectContainer
def is_binary(content):
    """See if the first few bytes contain any null characters.

    Args:
      content: Bytestring to check for binary content
    """
    return b'\x00' in content[:FIRST_FEW_BYTES]