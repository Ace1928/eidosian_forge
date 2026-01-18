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
def test_ex_import_snapshot(self):
    disk_container = [{'Description': 'Dummy import snapshot task', 'Format': 'raw', 'UserBucket': {'S3Bucket': 'dummy-bucket', 'S3Key': 'dummy-key'}}]
    snap = self.driver.ex_import_snapshot(disk_container=disk_container)
    self.assertEqual(snap.id, 'snap-0ea83e8a87e138f39')