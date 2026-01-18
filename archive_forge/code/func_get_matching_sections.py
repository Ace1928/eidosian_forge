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
def get_matching_sections(self, name):
    store = self.get_store(self)
    store._load_from_string(b'\n[foo]\noption=foo\n[foo/baz]\noption=foo/baz\n[bar]\noption=bar\n')
    matcher = self.matcher(store, name)
    return list(matcher.get_sections())