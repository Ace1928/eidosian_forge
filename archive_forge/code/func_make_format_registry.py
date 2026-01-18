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
def make_format_registry(self):
    my_format_registry = controldir.ControlDirFormatRegistry()
    my_format_registry.register('deprecated', DeprecatedBzrDirFormat, 'Some format.  Slower and unawesome and deprecated.', deprecated=True)
    my_format_registry.register_lazy('lazy', __name__, 'DeprecatedBzrDirFormat', 'Format registered lazily', deprecated=True)
    bzr.register_metadir(my_format_registry, 'knit', 'breezy.bzr.knitrepo.RepositoryFormatKnit1', 'Format using knits')
    my_format_registry.set_default('knit')
    bzr.register_metadir(my_format_registry, 'branch6', 'breezy.bzr.knitrepo.RepositoryFormatKnit3', 'Experimental successor to knit.  Use at your own risk.', branch_format='breezy.bzr.branch.BzrBranchFormat6', experimental=True)
    bzr.register_metadir(my_format_registry, 'hidden format', 'breezy.bzr.knitrepo.RepositoryFormatKnit3', 'Experimental successor to knit.  Use at your own risk.', branch_format='breezy.bzr.branch.BzrBranchFormat6', hidden=True)
    my_format_registry.register('hiddendeprecated', DeprecatedBzrDirFormat, 'Old format.  Slower and does not support things. ', hidden=True)
    my_format_registry.register_lazy('hiddenlazy', __name__, 'DeprecatedBzrDirFormat', 'Format registered lazily', deprecated=True, hidden=True)
    return my_format_registry