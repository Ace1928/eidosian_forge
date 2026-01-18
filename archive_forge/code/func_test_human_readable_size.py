import collections
import http.client as http
import io
from unittest import mock
import copy
import os
import sys
import uuid
import fixtures
from oslo_serialization import jsonutils
import webob
from glance.cmd import replicator as glance_replicator
from glance.common import exception
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def test_human_readable_size(self):
    _human_readable_size = glance_replicator._human_readable_size
    self.assertEqual('0.0 B', _human_readable_size(0))
    self.assertEqual('1.0 B', _human_readable_size(1))
    self.assertEqual('512.0 B', _human_readable_size(512))
    self.assertEqual('1.0 KiB', _human_readable_size(1024))
    self.assertEqual('2.0 KiB', _human_readable_size(2048))
    self.assertEqual('8.0 KiB', _human_readable_size(8192))
    self.assertEqual('64.0 KiB', _human_readable_size(65536))
    self.assertEqual('93.3 KiB', _human_readable_size(95536))
    self.assertEqual('117.7 MiB', _human_readable_size(123456789))
    self.assertEqual('36.3 GiB', _human_readable_size(39022543360))