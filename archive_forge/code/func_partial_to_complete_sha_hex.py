from git.util import bin_to_hex, hex_to_bin
from gitdb.base import OInfo, OStream
from gitdb.db import GitDB
from gitdb.db import LooseObjectDB
from gitdb.exc import BadObject
from git.exc import GitCommandError
from typing import TYPE_CHECKING
from git.types import PathLike
def partial_to_complete_sha_hex(self, partial_hexsha: str) -> bytes:
    """
        :return: Full binary 20 byte sha from the given partial hexsha

        :raise AmbiguousObjectName:
        :raise BadObject:

        :note: Currently we only raise :class:`BadObject` as git does not communicate
            AmbiguousObjects separately.
        """
    try:
        hexsha, _typename, _size = self._git.get_object_header(partial_hexsha)
        return hex_to_bin(hexsha)
    except (GitCommandError, ValueError) as e:
        raise BadObject(partial_hexsha) from e