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
def test_invalid_content(self):
    store = config.TransportIniFileStore(self.get_transport(), 'foo.conf')
    self.assertEqual(False, store.is_loaded())
    exc = self.assertRaises(config.ParseConfigError, store._load_from_string, b'this is invalid !')
    self.assertEndsWith(exc.filename, 'foo.conf')
    self.assertEqual(False, store.is_loaded())