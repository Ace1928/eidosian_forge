import json
import copy
import tempfile
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.layer1 import Layer1
from boto.compat import six
def test_describe_vaults(self):
    content = copy.copy(self.vault_info)
    content[u'RequestId'] = None
    self.set_http_response(status_code=200, header=self.json_header, body=json.dumps(content).encode('utf-8'))
    api_response = self.service_connection.describe_vault(self.vault_name)
    self.assertDictEqual(content, api_response)