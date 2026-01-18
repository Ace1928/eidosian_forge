import os
import tarfile
import unittest
import warnings
import zipfile
from os.path import join
from textwrap import dedent
from test.support import captured_stdout
from test.support.warnings_helper import check_warnings
from distutils.command.sdist import sdist, show_formats
from distutils.core import Distribution
from distutils.tests.test_config import BasePyPIRCCommandTestCase
from distutils.errors import DistutilsOptionError
from distutils.spawn import find_executable
from distutils.log import WARN
from distutils.filelist import FileList
from distutils.archive_util import ARCHIVE_FORMATS
from distutils.core import setup
import somecode
@unittest.skipUnless(ZLIB_SUPPORT, 'Need zlib support to run')
def test_add_defaults(self):
    dist, cmd = self.get_cmd()
    dist.package_data = {'': ['*.cfg', '*.dat'], 'somecode': ['*.txt']}
    self.write_file((self.tmp_dir, 'somecode', 'doc.txt'), '#')
    self.write_file((self.tmp_dir, 'somecode', 'doc.dat'), '#')
    data_dir = join(self.tmp_dir, 'data')
    os.mkdir(data_dir)
    self.write_file((data_dir, 'data.dt'), '#')
    some_dir = join(self.tmp_dir, 'some')
    os.mkdir(some_dir)
    hg_dir = join(self.tmp_dir, '.hg')
    os.mkdir(hg_dir)
    self.write_file((hg_dir, 'last-message.txt'), '#')
    self.write_file((self.tmp_dir, 'buildout.cfg'), '#')
    self.write_file((self.tmp_dir, 'inroot.txt'), '#')
    self.write_file((some_dir, 'file.txt'), '#')
    self.write_file((some_dir, 'other_file.txt'), '#')
    dist.data_files = [('data', ['data/data.dt', 'buildout.cfg', 'inroot.txt', 'notexisting']), 'some/file.txt', 'some/other_file.txt']
    script_dir = join(self.tmp_dir, 'scripts')
    os.mkdir(script_dir)
    self.write_file((script_dir, 'script.py'), '#')
    dist.scripts = [join('scripts', 'script.py')]
    cmd.formats = ['zip']
    cmd.use_defaults = True
    cmd.ensure_finalized()
    cmd.run()
    dist_folder = join(self.tmp_dir, 'dist')
    files = os.listdir(dist_folder)
    self.assertEqual(files, ['fake-1.0.zip'])
    zip_file = zipfile.ZipFile(join(dist_folder, 'fake-1.0.zip'))
    try:
        content = zip_file.namelist()
    finally:
        zip_file.close()
    expected = ['', 'PKG-INFO', 'README', 'buildout.cfg', 'data/', 'data/data.dt', 'inroot.txt', 'scripts/', 'scripts/script.py', 'setup.py', 'some/', 'some/file.txt', 'some/other_file.txt', 'somecode/', 'somecode/__init__.py', 'somecode/doc.dat', 'somecode/doc.txt']
    self.assertEqual(sorted(content), ['fake-1.0/' + x for x in expected])
    f = open(join(self.tmp_dir, 'MANIFEST'))
    try:
        manifest = f.read()
    finally:
        f.close()
    self.assertEqual(manifest, MANIFEST % {'sep': os.sep})