import hashlib
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional
from pip._internal.exceptions import HashMismatch, HashMissing, InstallationError
from pip._internal.utils.misc import read_chunks
def check_against_file(self, file: BinaryIO) -> None:
    """Check good hashes against a file-like object

        Raise HashMismatch if none match.

        """
    return self.check_against_chunks(read_chunks(file))