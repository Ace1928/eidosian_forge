import hashlib
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional
from pip._internal.exceptions import HashMismatch, HashMissing, InstallationError
from pip._internal.utils.misc import read_chunks
def check_against_path(self, path: str) -> None:
    with open(path, 'rb') as file:
        return self.check_against_file(file)