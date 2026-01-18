import glob
import operator
import os
import shutil
import sys
import tempfile
from incremental import Version
from twisted.python import release
from twisted.python._release import (
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
def makeProject(self, version, baseDirectory=None):
    """
        Make a Twisted-style project in the given base directory.

        @param baseDirectory: The directory to create files in
            (as a L{FilePath).
        @param version: The version information for the project.
        @return: L{Project} pointing to the created project.
        """
    if baseDirectory is None:
        baseDirectory = FilePath(self.mktemp())
    segments = version[0].split('.')
    directory = baseDirectory
    for segment in segments:
        directory = directory.child(segment)
        if not directory.exists():
            directory.createDirectory()
        directory.child('__init__.py').setContent(b'')
    directory.child('newsfragments').createDirectory()
    directory.child('_version.py').setContent(genVersion(*version).encode())
    return Project(directory)