import os
from filelock import FileLock as FileLock_
from filelock import UnixFileLock
from filelock import __version__ as _filelock_version
from packaging import version
@classmethod
def hash_filename_if_too_long(cls, path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    filename = os.path.basename(path)
    max_filename_length = cls.MAX_FILENAME_LENGTH
    if issubclass(cls, UnixFileLock):
        max_filename_length = min(max_filename_length, os.statvfs(os.path.dirname(path)).f_namemax)
    if len(filename) > max_filename_length:
        dirname = os.path.dirname(path)
        hashed_filename = str(hash(filename))
        new_filename = filename[:max_filename_length - len(hashed_filename) - 8] + '...' + hashed_filename + '.lock'
        return os.path.join(dirname, new_filename)
    else:
        return path