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
def assertSectionIDs(self, expected, location, content):
    self.store._load_from_string(content)
    matcher = config.StartingPathMatcher(self.store, location)
    sections = list(matcher.get_sections())
    self.assertLength(len(expected), sections)
    self.assertEqual(expected, [section.id for _, section in sections])
    return sections