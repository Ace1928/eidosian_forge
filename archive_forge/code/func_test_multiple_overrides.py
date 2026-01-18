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
def test_multiple_overrides(self):
    self.store._from_cmdline(['a=b', 'x=y'])
    section = self.get_section()
    self.assertEqual('b', section.get('a'))
    self.assertEqual('y', section.get('x'))