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
def test_locals_dont_leak(self):
    """Make sure we chose the right local in presence of several sections.
        """
    l_store = config.LocationStore()
    l_store._load_from_string(b'\n[/home/user]\nlfoo = loc-foo/{relpath}\n[/home/user/project]\nlfoo = loc-foo/{relpath}\n')
    l_store.save()
    stack = config.LocationStack('/home/user/project/branch')
    self.assertEqual('loc-foo/branch', stack.get('lfoo', expand=True))
    stack = config.LocationStack('/home/user/bar/baz')
    self.assertEqual('loc-foo/bar/baz', stack.get('lfoo', expand=True))