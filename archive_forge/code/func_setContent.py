from __future__ import annotations
import base64
import errno
import os
import sys
from os import listdir, stat, utime
from os.path import (
from stat import (
from typing import (
from zope.interface import Attribute, Interface, implementer
from typing_extensions import Literal
from twisted.python.compat import cmp, comparable
from twisted.python.runtime import platform
from twisted.python.util import FancyEqMixin
from twisted.python.win32 import (
def setContent(self, content: bytes, ext: Union[str, bytes]='.new') -> None:
    """
        Replace the file at this path with a new file that contains the given
        bytes, trying to avoid data-loss in the meanwhile.

        On UNIX-like platforms, this method does its best to ensure that by the
        time this method returns, either the old contents I{or} the new
        contents of the file will be present at this path for subsequent
        readers regardless of premature device removal, program crash, or power
        loss, making the following assumptions:

            - your filesystem is journaled (i.e. your filesystem will not
              I{itself} lose data due to power loss)

            - your filesystem's C{rename()} is atomic

            - your filesystem will not discard new data while preserving new
              metadata (see U{http://mjg59.livejournal.com/108257.html} for
              more detail)

        On most versions of Windows there is no atomic C{rename()} (see
        U{http://bit.ly/win32-overwrite} for more information), so this method
        is slightly less helpful.  There is a small window where the file at
        this path may be deleted before the new file is moved to replace it:
        however, the new file will be fully written and flushed beforehand so
        in the unlikely event that there is a crash at that point, it should be
        possible for the user to manually recover the new version of their
        data.  In the future, Twisted will support atomic file moves on those
        versions of Windows which I{do} support them: see U{Twisted ticket
        3004<http://twistedmatrix.com/trac/ticket/3004>}.

        This method should be safe for use by multiple concurrent processes,
        but note that it is not easy to predict which process's contents will
        ultimately end up on disk if they invoke this method at close to the
        same time.

        @param content: The desired contents of the file at this path.
        @type content: L{bytes}

        @param ext: An extension to append to the temporary filename used to
            store the bytes while they are being written.  This can be used to
            make sure that temporary files can be identified by their suffix,
            for cleanup in case of crashes.
        @type ext: L{bytes}
        """
    sib = self.temporarySibling(ext)
    with sib.open('w') as f:
        f.write(content)
    if platform.isWindows() and exists(self.path):
        os.unlink(self.path)
    os.rename(sib.path, self.asBytesMode().path)