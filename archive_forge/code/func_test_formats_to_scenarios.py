import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
def test_formats_to_scenarios(self):
    from .per_repository import formats_to_scenarios
    formats = [('(c)', remote.RemoteRepositoryFormat()), ('(d)', repository.format_registry.get(b'Bazaar repository format 2a (needs bzr 1.16 or later)\n'))]
    no_vfs_scenarios = formats_to_scenarios(formats, 'server', 'readonly', None)
    vfs_scenarios = formats_to_scenarios(formats, 'server', 'readonly', vfs_transport_factory='vfs')
    expected = [('RemoteRepositoryFormat(c)', {'bzrdir_format': remote.RemoteBzrDirFormat(), 'repository_format': remote.RemoteRepositoryFormat(), 'transport_readonly_server': 'readonly', 'transport_server': 'server'}), ('RepositoryFormat2a(d)', {'bzrdir_format': bzrdir.BzrDirMetaFormat1(), 'repository_format': groupcompress_repo.RepositoryFormat2a(), 'transport_readonly_server': 'readonly', 'transport_server': 'server'})]
    self.assertEqual(expected, no_vfs_scenarios)
    self.assertEqual([('RemoteRepositoryFormat(c)', {'bzrdir_format': remote.RemoteBzrDirFormat(), 'repository_format': remote.RemoteRepositoryFormat(), 'transport_readonly_server': 'readonly', 'transport_server': 'server', 'vfs_transport_factory': 'vfs'}), ('RepositoryFormat2a(d)', {'bzrdir_format': bzrdir.BzrDirMetaFormat1(), 'repository_format': groupcompress_repo.RepositoryFormat2a(), 'transport_readonly_server': 'readonly', 'transport_server': 'server', 'vfs_transport_factory': 'vfs'})], vfs_scenarios)