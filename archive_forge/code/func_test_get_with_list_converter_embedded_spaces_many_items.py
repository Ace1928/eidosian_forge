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
def test_get_with_list_converter_embedded_spaces_many_items(self):
    self.register_list_option('foo', None)
    conf = self.get_conf(b'foo=" bar", "baz "')
    self.assertEqual([' bar', 'baz '], conf.get('foo'))