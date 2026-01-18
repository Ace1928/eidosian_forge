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
def test_fallback_first_wins(self):
    r = config.CredentialStoreRegistry()
    stub1 = StubCredentialStore()
    stub1.add_credentials('http', 'example.com', 'somebody', 'stub1')
    r.register('stub1', stub1, fallback=True)
    stub2 = StubCredentialStore()
    stub2.add_credentials('http', 'example.com', 'somebody', 'stub2')
    r.register('stub2', stub1, fallback=True)
    creds = r.get_fallback_credentials('http', 'example.com')
    self.assertEqual('somebody', creds['user'])
    self.assertEqual('stub1', creds['password'])