import os
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Dict
from zope.interface import Interface, implementer
from twisted.python.compat import execfile
@staticmethod
def exportTo(fromDir, exportDir):
    """
        Export the content of a Git repository to the specified directory.

        @type fromDir: L{twisted.python.filepath.FilePath}
        @param fromDir: The path to the Git repository to export.

        @type exportDir: L{twisted.python.filepath.FilePath}
        @param exportDir: The directory to export the content of the
            repository to. This directory doesn't have to exist prior to
            exporting the repository.
        """
    runCommand(['git', '-C', fromDir.path, 'checkout-index', '--all', '--force', '--prefix', exportDir.path + '/'])