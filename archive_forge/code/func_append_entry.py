from mmap import mmap
import re
import time as _time
from git.compat import defenc
from git.objects.util import (
from git.util import (
import os.path as osp
from typing import Iterator, List, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
@classmethod
def append_entry(cls, config_reader: Union[Actor, 'GitConfigParser', 'SectionConstraint', None], filepath: PathLike, oldbinsha: bytes, newbinsha: bytes, message: str, write: bool=True) -> 'RefLogEntry':
    """Append a new log entry to the revlog at filepath.

        :param config_reader: Configuration reader of the repository - used to obtain
            user information. May also be an Actor instance identifying the committer
            directly or None.
        :param filepath: Full path to the log file.
        :param oldbinsha: Binary sha of the previous commit.
        :param newbinsha: Binary sha of the current commit.
        :param message: Message describing the change to the reference.
        :param write: If True, the changes will be written right away. Otherwise the
            change will not be written.

        :return: RefLogEntry objects which was appended to the log.

        :note: As we are append-only, concurrent access is not a problem as we do not
            interfere with readers.
        """
    if len(oldbinsha) != 20 or len(newbinsha) != 20:
        raise ValueError('Shas need to be given in binary format')
    assure_directory_exists(filepath, is_file=True)
    first_line = message.split('\n')[0]
    if isinstance(config_reader, Actor):
        committer = config_reader
    else:
        committer = Actor.committer(config_reader)
    entry = RefLogEntry((bin_to_hex(oldbinsha).decode('ascii'), bin_to_hex(newbinsha).decode('ascii'), committer, (int(_time.time()), _time.altzone), first_line))
    if write:
        lf = LockFile(filepath)
        lf._obtain_lock_or_raise()
        fd = open(filepath, 'ab')
        try:
            fd.write(entry.format().encode(defenc))
        finally:
            fd.close()
            lf._release_lock()
    return entry