import os
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Dict
from zope.interface import Interface, implementer
from twisted.python.compat import execfile
@staticmethod
def ensureIsWorkingDirectory(path):
    """
        Ensure that C{path} is a Git working directory.

        @type path: L{twisted.python.filepath.FilePath}
        @param path: The path to check.
        """
    try:
        runCommand(['git', 'rev-parse'], cwd=path.path)
    except (CalledProcessError, OSError):
        raise NotWorkingDirectory(f'{path.path} does not appear to be a Git repository.')