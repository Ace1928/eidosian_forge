import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, Connection, XmlResponse, JsonResponse
from libcloud.test.file_fixtures import ComputeFileFixtures
def test_json_unicode_response(self):
    self.connection.responseCls = JsonResponse
    r = self.connection.request('/unicode/json')
    self.assertEqual(r.object, {'test': 'Ś'})