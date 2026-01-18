import io
import re
import sys
from unittest import mock
import fixtures
from keystoneauth1 import fixture
from testtools import matchers
from zunclient import api_versions
from zunclient import exceptions
import zunclient.shell
from zunclient.tests.unit import utils
def register_keystone_discovery_fixture(self, mreq):
    v3_url = 'http://no.where/v3'
    v3_version = fixture.V3Discovery(v3_url)
    mreq.register_uri('GET', v3_url, json=_create_ver_list([v3_version]), status_code=200)