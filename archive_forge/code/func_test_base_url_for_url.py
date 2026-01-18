import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_base_url_for_url(self):
    self.assertEqual('file:///foo/bar', utils.base_url_for_url('file:///foo/bar/baz'))
    self.assertEqual('file:///foo/bar', utils.base_url_for_url('file:///foo/bar/baz.txt'))
    self.assertEqual('file:///foo/bar', utils.base_url_for_url('file:///foo/bar/'))
    self.assertEqual('file:///', utils.base_url_for_url('file:///'))
    self.assertEqual('file:///', utils.base_url_for_url('file:///foo'))
    self.assertEqual('http://foo/bar', utils.base_url_for_url('http://foo/bar/'))
    self.assertEqual('http://foo/bar', utils.base_url_for_url('http://foo/bar/baz.template'))