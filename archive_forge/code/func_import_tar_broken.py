import os
import tarfile
import tempfile
import warnings
from io import BytesIO
from shutil import copy2, copytree, rmtree
from .. import osutils
from .. import revision as _mod_revision
from .. import transform
from ..controldir import ControlDir
from ..export import export
from ..upstream_import import (NotArchiveType, ZipFileWrapper,
from . import TestCaseInTempDir, TestCaseWithTransport
from .features import UnicodeFilenameFeature
def import_tar_broken(tree, tar_input):
    """
    Import a tarfile with names that that end in //, e.g. Feisty Python 2.5
    """
    tar_file = tarfile.open('lala', 'r', tar_input)
    for member in tar_file.members:
        if member.name.endswith('/'):
            member.name += '/'
    import_archive(tree, tar_file)