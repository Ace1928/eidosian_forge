from io import BytesIO
import breezy.bzr.bzrdir
import breezy.mergeable
import breezy.transport
import breezy.urlutils
from ... import errors, tests
from ...tests.per_transport import transport_test_permutations
from ...tests.scenarios import load_tests_apply_scenarios
from ...tests.test_transport import TestTransportImplementation
from ..bundle.serializer import write_bundle
def read_mergeable_from_url(self, url):
    return breezy.mergeable.read_mergeable_from_url(url, possible_transports=self.possible_transports)