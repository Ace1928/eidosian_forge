import os
import unittest
import time
from boto.swf.layer1 import Layer1
from boto.swf import exceptions as swf_exceptions
def test_list_domains(self):
    r = self.conn.list_domains('REGISTERED')
    found = None
    for info in r['domainInfos']:
        if info['name'] == self._domain:
            found = info
            break
    self.assertNotEqual(found, None, 'list_domains; test domain not found')
    self.assertEqual(found['description'], self._domain_description, 'list_domains; description does not match')
    self.assertEqual(found['status'], 'REGISTERED', 'list_domains; status does not match')