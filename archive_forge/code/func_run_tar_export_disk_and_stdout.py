import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def run_tar_export_disk_and_stdout(self, extension, tarfile_flags):
    tree = self.make_basic_tree()
    fname = 'test.{}'.format(extension)
    self.run_bzr('export -d tree {}'.format(fname))
    mode = 'r|{}'.format(tarfile_flags)
    with tarfile.open(fname, mode=mode) as ball:
        self.assertTarANameAndContent(ball, root='test/')
    content = self.run_bzr_raw('export -d tree --format={} -'.format(extension))[0]
    with tarfile.open(mode=mode, fileobj=BytesIO(content)) as ball:
        self.assertTarANameAndContent(ball, root='')