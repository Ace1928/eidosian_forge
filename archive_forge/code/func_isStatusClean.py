import os
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Dict
from zope.interface import Interface, implementer
from twisted.python.compat import execfile
@staticmethod
def isStatusClean(path):
    """
        Return the Git status of the files in the specified path.

        @type path: L{twisted.python.filepath.FilePath}
        @param path: The path to get the status from (can be a directory or a
            file.)
        """
    status = runCommand(['git', '-C', path.path, 'status', '--short']).strip()
    return status == b''