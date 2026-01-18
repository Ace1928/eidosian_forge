from keystoneauth1 import exceptions as ksa_exceptions
import testresources
from testtools import matchers
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils as test_utils
from keystoneclient import utils
def test_find_by_int_name(self):
    output = utils.find_resource(self.manager, 9876)
    self.assertEqual(output, self.manager.resources['5678'])