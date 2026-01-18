from binascii import b2a_hex
from pathlib import Path
from .util import pack, unpack
from git.objects import Blob
from typing import NamedTuple, Sequence, TYPE_CHECKING, Tuple, Union, cast, List
from git.types import PathLike
@classmethod
def from_blob(cls, blob: Blob, stage: int=0) -> 'IndexEntry':
    """:return: Minimal entry resembling the given blob object"""
    time = pack('>LL', 0, 0)
    return IndexEntry((blob.mode, blob.binsha, stage << CE_STAGESHIFT, blob.path, time, time, 0, 0, 0, 0, blob.size))