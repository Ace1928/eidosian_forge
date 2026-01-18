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
def test_find_merge_tool_not_found(self):
    conf = self._get_sample_config()
    cmdline = conf.find_merge_tool('DOES NOT EXIST')
    self.assertIs(cmdline, None)