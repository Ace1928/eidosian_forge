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
def archive_test(self, builder, importer, subdir=False):
    archive_file = self.make_archive(builder, subdir)
    tree = ControlDir.create_standalone_workingtree('tree')
    with tree.lock_write():
        importer(tree, archive_file)
        self.assertTrue(tree.is_versioned('README'))
        self.assertTrue(tree.is_versioned('FEEDME'))
        self.assertTrue(os.path.isfile(tree.abspath('README')))
        self.assertEqual(tree.stored_kind('README'), 'file')
        self.assertEqual(tree.stored_kind('FEEDME'), 'file')
        with open(tree.abspath('junk/food'), 'wb') as f:
            f.write(b'I like food\n')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            archive_file = self.make_archive2(builder, subdir)
            importer(tree, archive_file)
        self.assertTrue(tree.is_versioned('README'))
        self.assertEqual(tree.get_file_text('README'), b'Wow?')
        self.assertTrue(not os.path.exists(tree.abspath('FEEDME')))