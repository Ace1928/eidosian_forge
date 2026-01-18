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
def test_glob_included(self):
    sections = self.assertSectionIDs(['/foo/*/baz', '/foo/b*', '/foo'], '/foo/bar/baz', b'[/foo]\n[/foo/qux]\n[/foo/b*]\n[/foo/*/baz]\n')
    self.assertEqual(['', 'baz', 'bar/baz'], [s.locals['relpath'] for _, s in sections])