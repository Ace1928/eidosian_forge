import io
import sys
from unittest import mock
from urllib import parse
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils as test_utils
from novaclient import utils
from novaclient.v2 import servers
def test_find_without_wrapping_exception(self):
    alphanum_manager = FakeManager(True)
    self.assertRaises(exceptions.NotFound, utils.find_resource, alphanum_manager, 'not_exist', wrap_exception=False)
    res = alphanum_manager.resources[0]
    alphanum_manager.resources.append(res)
    self.assertRaises(exceptions.NoUniqueMatch, utils.find_resource, alphanum_manager, res.name, wrap_exception=False)