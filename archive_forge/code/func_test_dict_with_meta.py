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
def test_dict_with_meta(self):
    resp = create_response_obj_with_header()
    obj = common_base.DictWithMeta([], resp)
    self.assertEqual({}, obj)
    self.assertTrue(hasattr(obj, 'request_ids'))
    self.assertEqual([REQUEST_ID], obj.request_ids)