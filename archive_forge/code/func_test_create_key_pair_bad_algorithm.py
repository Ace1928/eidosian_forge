import os
import uuid
from oslo_config import cfg
from oslo_utils import uuidutils
from oslotest import base
import requests
from testtools import testcase
from castellan.common import exception
from castellan.key_manager import vault_key_manager
from castellan.tests.functional import config
from castellan.tests.functional.key_manager import test_key_manager
def test_create_key_pair_bad_algorithm(self):
    self.assertRaises(NotImplementedError, self.key_mgr.create_key_pair, self.ctxt, 'DSA', 2048)