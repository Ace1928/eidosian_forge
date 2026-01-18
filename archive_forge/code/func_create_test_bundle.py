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
def create_test_bundle(self):
    out, wt = create_bundle_file(self)
    if self.get_transport().is_readonly():
        self.build_tree_contents([(self.bundle_name, out.getvalue())])
    else:
        self.get_transport().put_file(self.bundle_name, out)
        self.log('Put to: %s', self.get_url(self.bundle_name))
    return wt