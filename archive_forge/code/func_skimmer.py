from distutils import log, dir_util
import os
from setuptools import Command
from setuptools import namespaces
from setuptools.archive_util import unpack_archive
from .._path import ensure_directory
def skimmer(src, dst):
    for skip in ('.svn/', 'CVS/'):
        if src.startswith(skip) or '/' + skip in src:
            return None
    self.outputs.append(dst)
    log.debug('Copying %s to %s', src, dst)
    return dst