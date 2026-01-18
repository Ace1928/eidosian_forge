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
def test_more_specific_sections_first(self):
    store = self.get_store(self)
    store._load_from_string(b'\n[/foo]\nsection=/foo\n[/foo/bar]\nsection=/foo/bar\n')
    self.assertEqual(['/foo', '/foo/bar'], [section.id for _, section in store.get_sections()])
    matcher = config.LocationMatcher(store, '/foo/bar/baz')
    sections = [section for _, section in matcher.get_sections()]
    self.assertEqual(['/foo/bar', '/foo'], [section.id for section in sections])
    self.assertEqual(['baz', 'bar/baz'], [section.extra_path for section in sections])