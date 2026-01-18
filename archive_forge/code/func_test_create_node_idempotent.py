import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
def test_create_node_idempotent(self):
    EC2MockHttp.type = 'idempotent'
    image = NodeImage(id='ami-be3adfd7', name=self.image_name, driver=self.driver)
    size = NodeSize('m1.small', 'Small Instance', None, None, None, None, driver=self.driver)
    token = 'testclienttoken'
    node = self.driver.create_node(name='foo', image=image, size=size, ex_clienttoken=token)
    self.assertEqual(node.id, 'i-2ba64342')
    self.assertEqual(node.extra['client_token'], token)
    EC2MockHttp.type = 'idempotent_mismatch'
    idem_error = None
    try:
        self.driver.create_node(name='foo', image=image, size=size, ex_mincount='2', ex_maxcount='2', ex_clienttoken=token)
    except IdempotentParamError as e:
        idem_error = e
    self.assertTrue(idem_error is not None)