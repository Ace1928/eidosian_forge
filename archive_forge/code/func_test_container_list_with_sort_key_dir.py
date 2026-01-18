import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_container_list_with_sort_key_dir(self):
    expect = [('GET', '/v1/containers/?sort_key=uuid&sort_dir=desc', {}, None)]
    self._test_containers_list_with_filters(sort_key='uuid', sort_dir='desc', expect=expect)