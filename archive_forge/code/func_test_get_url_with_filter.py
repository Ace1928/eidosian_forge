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
def test_get_url_with_filter(self):
    url = '/fake'
    for case in self.cases:
        self.assertEqual('%s%s' % (url, case[1]), parse.unquote_plus(utils.get_url_with_filter(url, case[0])))