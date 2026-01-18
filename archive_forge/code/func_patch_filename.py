import email.parser
import time
from difflib import SequenceMatcher
from typing import BinaryIO, Optional, TextIO, Union
from .objects import S_ISGITLINK, Blob, Commit
from .pack import ObjectContainer
def patch_filename(p, root):
    if p is None:
        return b'/dev/null'
    else:
        return root + b'/' + p