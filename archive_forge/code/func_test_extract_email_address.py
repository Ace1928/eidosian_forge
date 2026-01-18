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
def test_extract_email_address(self):
    self.assertEqual('jane@test.com', config.extract_email_address('Jane <jane@test.com>'))
    self.assertRaises(config.NoEmailInUsername, config.extract_email_address, 'Jane Tester')