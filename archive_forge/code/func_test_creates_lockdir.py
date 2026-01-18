import sys
from io import BytesIO
from stat import S_ISDIR
from ...bzr.bzrdir import BzrDirMetaFormat1
from ...bzr.serializer import format_registry as serializer_format_registry
from ...errors import IllegalPath
from ...repository import InterRepository, Repository
from ...tests import TestCase, TestCaseWithTransport
from ...transport import NoSuchFile
from . import xml4
from .bzrdir import BzrDirFormat6
from .repository import (InterWeaveRepo, RepositoryFormat4, RepositoryFormat5,
def test_creates_lockdir(self):
    """Make sure it appears to be controlled by a LockDir existence"""
    control = BzrDirMetaFormat1().initialize(self.get_url())
    repo = RepositoryFormat7().initialize(control, shared=True)
    t = control.get_repository_transport(None)
    self.assertFalse(t.has('lock/held/info'))
    with repo.lock_write():
        self.assertTrue(t.has('lock/held/info'))