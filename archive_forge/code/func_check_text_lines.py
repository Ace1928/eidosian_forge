from itertools import chain
from .errors import BinaryFile
from .iterablefile import IterableFile
from .osutils import file_iterator
def check_text_lines(lines):
    """Raise BinaryFile if the supplied lines contain NULs.
    Only the first 1024 characters are checked.
    """
    f = IterableFile(lines)
    if b'\x00' in f.read(1024):
        raise BinaryFile()