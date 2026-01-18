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
def prepare_default_stacking(self, child_format='1.6'):
    parent_bzrdir = self.make_controldir('.')
    child_branch = self.make_branch('child', format=child_format)
    parent_bzrdir.get_config().set_default_stack_on(child_branch.base)
    new_child_transport = parent_bzrdir.transport.clone('child2')
    return (child_branch, new_child_transport)