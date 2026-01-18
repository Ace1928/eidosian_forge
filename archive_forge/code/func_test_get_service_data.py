import copy
import datetime
import os
import tempfile
from oslotest import base
import os_service_types.service_types
def test_get_service_data(self):
    service_data = self.service_types.get_service_data(self.service_type)
    api_url = 'https://developer.openstack.org/api-ref/{api_reference}/'
    if not self.official:
        self.assertIsNone(service_data)
    else:
        self.assertIsNotNone(service_data)
        self.assertEqual(self.project, service_data['project'])
        self.assertEqual(self.official, service_data['service_type'])
        self.assertEqual(api_url.format(api_reference=self.api_reference), service_data['api_reference'])