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
def test_config_creates_local(self):
    """Creating a new entry in config uses a local path."""
    branch = self.make_branch('branch', format='knit')
    branch.set_push_location('http://foobar')
    local_path = osutils.getcwd().encode('utf8')
    self.check_file_contents(bedding.locations_config_path(), b'[%s/branch]\npush_location = http://foobar\npush_location:policy = norecurse\n' % (local_path,))