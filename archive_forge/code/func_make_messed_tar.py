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
def make_messed_tar(self):
    result = BytesIO()
    with tarfile.open('project-0.1.tar', 'w', result) as tar_file:
        os.mkdir('project-0.1')
        tar_file.add('project-0.1')
        os.mkdir('project-0.2')
        tar_file.add('project-0.2')
        with open('project-0.1/README', 'wb') as f:
            f.write(b'What?')
        tar_file.add('project-0.1/README')
    rmtree('project-0.1')
    result.seek(0)
    return result