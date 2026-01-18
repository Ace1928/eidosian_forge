from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import hypervisors as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def compare_to_expected(self, expected, hyper):
    for key, value in expected.items():
        self.assertEqual(getattr(hyper, key), value)