from __future__ import unicode_literals, absolute_import
import sys
import abc
import errno
import select
import six
def fd_to_int(fd):
    assert isinstance(fd, int) or hasattr(fd, 'fileno')
    if isinstance(fd, int):
        return fd
    else:
        return fd.fileno()