from typing import Tuple
from pip._internal.utils.misc import splitext
def is_archive_file(name: str) -> bool:
    """Return True if `name` is a considered as an archive file."""
    ext = splitext(name)[1].lower()
    if ext in ARCHIVE_EXTENSIONS:
        return True
    return False