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
def test_dont_cross_unrelated_section(self):
    c = self.get_config('/another/branch/path', b'\n[/one/branch/path]\nfoo = hello\nbar = {foo}/2\n\n[/another/branch/path]\nbar = {foo}/2\n')
    self.assertRaises(config.ExpandingUnknownOption, c.get, 'bar', expand=True)