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
def test_add_non_ascii_attr_to_resource(self):
    info = {'gigabytes_тест': -1, 'volumes_тест': -1, 'id': 'admin'}
    res = base.Resource(None, info)
    for key, value in info.items():
        self.assertEqual(value, getattr(res, key, None))