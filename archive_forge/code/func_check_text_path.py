from itertools import chain
from .errors import BinaryFile
from .iterablefile import IterableFile
from .osutils import file_iterator
def check_text_path(path):
    """Check whether the supplied path is a text, not binary file.
    Raise BinaryFile if a NUL occurs in the first 1024 bytes.
    """
    with open(path, 'rb') as f:
        text_file(f)