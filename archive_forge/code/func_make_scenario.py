from breezy import branchbuilder, tests, transport, workingtree
from breezy.tests import per_controldir, test_server
from breezy.transport import memory
def make_scenario(transport_server, transport_readonly_server, workingtree_format):
    return {'transport_server': transport_server, 'transport_readonly_server': transport_readonly_server, 'bzrdir_format': workingtree_format._matchingcontroldir, 'workingtree_format': workingtree_format}