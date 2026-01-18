from keystoneauth1 import exceptions as ksa_exceptions
import testresources
from testtools import matchers
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils as test_utils
from keystoneclient import utils
def test_find_by_uuid(self):
    uuid = '8e8ec658-c7b0-4243-bdf8-6f7f2952c0d0'
    output = utils.find_resource(self.manager, uuid)
    self.assertEqual(output, self.manager.resources[uuid])