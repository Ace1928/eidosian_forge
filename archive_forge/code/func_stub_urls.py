import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def stub_urls():
    v2_m = self.stub_url('GET', ['v2.0'], base_url=self.BASE_URL, status_code=200, json={'version': v2_disc})
    common_m = self.stub_url('GET', base_url=self.BASE_URL, status_code=200, json=common_disc)
    return (v2_m, common_m)