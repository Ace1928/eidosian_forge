import copy
import functools
import random
import http.client
from oslo_serialization import jsonutils
from testtools import matchers as tt_matchers
import webob
from keystone.api import discovery
from keystone.common import json_home
from keystone.tests import unit
def test_json_home_root(self):
    exp_json_home_data = copy.deepcopy({'resources': V3_JSON_HOME_RESOURCES})
    json_home.translate_urls(exp_json_home_data, '/v3')
    self._test_json_home('/', exp_json_home_data)