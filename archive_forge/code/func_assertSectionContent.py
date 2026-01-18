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
def assertSectionContent(self, expected, store_and_section):
    """Assert that some options have the proper values in a section."""
    _, section = store_and_section
    expected_name, expected_options = expected
    self.assertEqual(expected_name, section.id)
    self.assertEqual(expected_options, {k: section.get(k) for k in expected_options.keys()})