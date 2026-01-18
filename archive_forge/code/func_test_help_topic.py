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
def test_help_topic(self):
    topics = help_topics.HelpTopicRegistry()
    registry = self.make_format_registry()
    topics.register('current-formats', registry.help_topic, 'Current formats')
    topics.register('other-formats', registry.help_topic, 'Other formats')
    new = topics.get_detail('current-formats')
    rest = topics.get_detail('other-formats')
    experimental, deprecated = rest.split('Deprecated formats')
    self.assertContainsRe(new, 'formats-help')
    self.assertContainsRe(new, ':knit:\n    \\(native\\) \\(default\\) Format using knits\n')
    self.assertContainsRe(experimental, ':branch6:\n    \\(native\\) Experimental successor to knit')
    self.assertContainsRe(deprecated, ':lazy:\n    \\(native\\) Format registered lazily\n')
    self.assertNotContainsRe(new, 'hidden')