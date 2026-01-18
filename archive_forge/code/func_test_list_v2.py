import re
from tempest.lib import exceptions
from glanceclient.tests.functional import base
def test_list_v2(self):
    out = self.glance('--os-image-api-version 2 image-list')
    endpoints = self.parser.listing(out)
    self.assertTableStruct(endpoints, ['ID', 'Name'])