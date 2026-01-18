import bisect
import codecs
import contextlib
import errno
import operator
import os
import stat
import sys
import time
import zlib
from stat import S_IEXEC
from .. import (cache_utf8, config, debug, errors, lock, osutils, trace,
from . import inventory, static_tuple
from .inventorytree import InventoryTreeChange
@classmethod
def on_file(cls, path, sha1_provider=None, worth_saving_limit=0, use_filesystem_for_exec=True):
    """Construct a DirState on the file at path "path".

        :param path: The path at which the dirstate file on disk should live.
        :param sha1_provider: an object meeting the SHA1Provider interface.
            If None, a DefaultSHA1Provider is used.
        :param worth_saving_limit: when the exact number of hash changed
            entries is known, only bother saving the dirstate if more than
            this count of entries have changed. -1 means never save.
        :param use_filesystem_for_exec: Whether to trust the filesystem
            for executable bit information
        :return: An unlocked DirState object, associated with the given path.
        """
    if sha1_provider is None:
        sha1_provider = DefaultSHA1Provider()
    result = cls(path, sha1_provider, worth_saving_limit=worth_saving_limit, use_filesystem_for_exec=use_filesystem_for_exec)
    return result