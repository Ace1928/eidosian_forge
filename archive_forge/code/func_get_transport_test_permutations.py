import os
import stat
import sys
from io import BytesIO
from .. import errors, osutils, pyutils, tests
from .. import transport as _mod_transport
from .. import urlutils
from ..errors import ConnectionError, PathError, TransportNotPossible
from ..osutils import getcwd
from ..transport import (ConnectedTransport, FileExists, NoSuchFile, Transport,
from ..transport.memory import MemoryTransport
from ..transport.remote import RemoteTransport
from . import TestNotApplicable, TestSkipped, multiply_tests, test_server
from .test_transport import TestTransportImplementation
def get_transport_test_permutations(module):
    """Get the permutations module wants to have tested."""
    if getattr(module, 'get_test_permutations', None) is None:
        raise AssertionError("transport module %s doesn't provide get_test_permutations()" % module.__name__)
        return []
    return module.get_test_permutations()