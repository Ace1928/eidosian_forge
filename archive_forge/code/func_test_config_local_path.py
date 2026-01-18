import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
def test_config_local_path(self):
    """The Branch.get_config will use a local system path"""
    branch = self.make_branch('branch')
    self.assertEqual('branch', branch.nick)
    local_path = osutils.getcwd().encode('utf8')
    config.LocationConfig.from_string(b'[%s/branch]\nnickname = barry' % (local_path,), 'branch', save=True)
    self.assertEqual('barry', branch.nick)