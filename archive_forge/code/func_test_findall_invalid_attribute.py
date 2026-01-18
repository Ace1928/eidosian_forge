from unittest import mock
import requests
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import exceptions
from cinderclient.tests.unit import test_utils
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import client
from cinderclient.v3 import volumes
def test_findall_invalid_attribute(self):
    cs.volumes.findall(vegetable='carrot')
    self.assertRaises(exceptions.NotFound, cs.volumes.find, vegetable='carrot')