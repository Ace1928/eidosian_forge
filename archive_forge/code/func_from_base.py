from binascii import b2a_hex
from pathlib import Path
from .util import pack, unpack
from git.objects import Blob
from typing import NamedTuple, Sequence, TYPE_CHECKING, Tuple, Union, cast, List
from git.types import PathLike
@classmethod
def from_base(cls, base: 'BaseIndexEntry') -> 'IndexEntry':
    """
        :return:
            Minimal entry as created from the given BaseIndexEntry instance.
            Missing values will be set to null-like values.

        :param base: Instance of type :class:`BaseIndexEntry`
        """
    time = pack('>LL', 0, 0)
    return IndexEntry((base.mode, base.binsha, base.flags, base.path, time, time, 0, 0, 0, 0, 0))