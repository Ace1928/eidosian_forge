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
def test_local_path_vs_url(self):
    self.assertSectionIDs(['file:///foo/bar', 'file:///foo'], '/foo/bar/baz', b'[file:///foo]\n[file:///foo/bar]\n')