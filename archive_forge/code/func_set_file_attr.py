import os
import errno
import sys
from hashlib import md5, sha1
from paramiko import util
from paramiko.sftp import (
from paramiko.sftp_si import SFTPServerInterface
from paramiko.sftp_attr import SFTPAttributes
from paramiko.common import DEBUG
from paramiko.server import SubsystemHandler
from paramiko.util import b
from paramiko.sftp import (
from paramiko.sftp_handle import SFTPHandle
@staticmethod
def set_file_attr(filename, attr):
    """
        Change a file's attributes on the local filesystem.  The contents of
        ``attr`` are used to change the permissions, owner, group ownership,
        and/or modification & access time of the file, depending on which
        attributes are present in ``attr``.

        This is meant to be a handy helper function for translating SFTP file
        requests into local file operations.

        :param str filename:
            name of the file to alter (should usually be an absolute path).
        :param .SFTPAttributes attr: attributes to change.
        """
    if sys.platform != 'win32':
        if attr._flags & attr.FLAG_PERMISSIONS:
            os.chmod(filename, attr.st_mode)
        if attr._flags & attr.FLAG_UIDGID:
            os.chown(filename, attr.st_uid, attr.st_gid)
    if attr._flags & attr.FLAG_AMTIME:
        os.utime(filename, (attr.st_atime, attr.st_mtime))
    if attr._flags & attr.FLAG_SIZE:
        with open(filename, 'w+') as f:
            f.truncate(attr.st_size)