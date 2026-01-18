import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
def test_contains_files(self):
    tempdir = self.mkdtemp()
    self.touch(os.path.join(tempdir, 'foo'))
    self.touch(os.path.join(tempdir, 'bar'))
    self.assertThat(tempdir, DirContains(['bar', 'foo']))