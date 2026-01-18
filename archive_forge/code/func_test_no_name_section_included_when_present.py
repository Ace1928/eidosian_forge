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
def test_no_name_section_included_when_present(self):
    sections = self.assertSectionIDs(['/foo/bar', '/foo', None], '/foo/bar/baz', b'option = defined so the no-name section exists\n[/foo]\n[/foo/bar]\n')
    self.assertEqual(['baz', 'bar/baz', '/foo/bar/baz'], [s.locals['relpath'] for _, s in sections])