from __future__ import unicode_literals, absolute_import
import sys
import abc
import errno
import select
import six
def select_fds(read_fds, timeout, selector=AutoSelector):
    """
    Wait for a list of file descriptors (`read_fds`) to become ready for
    reading. This chooses the most appropriate select-tool for use in
    prompt-toolkit.
    """
    fd_map = dict(((fd_to_int(fd), fd) for fd in read_fds))
    sel = selector()
    try:
        for fd in read_fds:
            sel.register(fd)
        result = sel.select(timeout)
        if result is not None:
            return [fd_map[fd_to_int(fd)] for fd in result]
    finally:
        sel.close()