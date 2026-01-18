import os
import subprocess
import sys
import breezy.branch
import breezy.bzr.branch
from ... import (branch, bzr, config, controldir, errors, help_topics, lock,
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ... import urlutils, win32utils
from ...errors import (NotBranchError, UnknownFormatError,
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from ...transport import memory, pathfilter
from ...transport.http.urllib import HttpTransport
from ...transport.nosmart import NoSmartTransportDecorator
from ...transport.readonly import ReadonlyTransportDecorator
from .. import branch as bzrbranch
from .. import (bzrdir, knitpack_repo, knitrepo, remote, workingtree_3,
from ..fullhistory import BzrBranchFormat5
def test_aliases(self):
    a_registry = controldir.ControlDirFormatRegistry()
    a_registry.register('deprecated', DeprecatedBzrDirFormat, 'Old format.  Slower and does not support stuff', deprecated=True)
    a_registry.register_alias('deprecatedalias', 'deprecated')
    self.assertEqual({'deprecatedalias': 'deprecated'}, a_registry.aliases())