from __future__ import annotations
import inspect
import sys
import warnings
import zipimport
from os.path import dirname, split as splitpath
from zope.interface import Interface, implementer
from twisted.python.compat import nativeString
from twisted.python.components import registerAdapter
from twisted.python.filepath import FilePath, UnlistableError
from twisted.python.reflect import namedAny
from twisted.python.zippath import ZipArchive
def mapPath(self, fsPathString):
    """
        Map the given FS path to a ZipPath, by looking at the ZipImporter's
        "archive" attribute and using it as our ZipArchive root, then walking
        down into the archive from there.

        @return: a L{zippath.ZipPath} or L{zippath.ZipArchive} instance.
        """
    za = ZipArchive(self.importer.archive)
    myPath = FilePath(self.importer.archive)
    itsPath = FilePath(fsPathString)
    if myPath == itsPath:
        return za
    segs = itsPath.segmentsFrom(myPath)
    zp = za
    for seg in segs:
        zp = zp.child(seg)
    return zp