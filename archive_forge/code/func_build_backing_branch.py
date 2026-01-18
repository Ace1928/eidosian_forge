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
def build_backing_branch(test, relpath, transport_class=None, server_class=None):
    """Test helper to create a backing branch only once.

    Some tests needs multiple stores/stacks to check concurrent update
    behaviours. As such, they need to build different branch *objects* even if
    they share the branch on disk.

    :param relpath: The relative path to the branch. (Note that the helper
        should always specify the same relpath).

    :param transport_class: The Transport class the test needs to use.

    :param server_class: The server associated with the ``transport_class``
        above.

    Either both or neither of ``transport_class`` and ``server_class`` should
    be specified.
    """
    if transport_class is not None and server_class is not None:
        test.transport_class = transport_class
        test.transport_server = server_class
    elif not (transport_class is None and server_class is None):
        raise AssertionError('Specify both ``transport_class`` and ``server_class`` or neither of them')
    if getattr(test, 'backing_branch', None) is None:
        test.backing_branch = test.make_branch(relpath)