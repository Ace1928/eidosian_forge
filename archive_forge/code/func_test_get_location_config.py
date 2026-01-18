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
def test_get_location_config(self):
    branch = FakeBranch()
    my_config = config.BranchConfig(branch)
    location_config = my_config._get_location_config()
    self.assertEqual(branch.base, location_config.location)
    self.assertIs(location_config, my_config._get_location_config())