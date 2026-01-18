import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_link_formatter(self):
    self.assertEqual('', utils.link_formatter(None))
    self.assertEqual('', utils.link_formatter([]))
    self.assertEqual('http://foo.example.com\nhttp://bar.example.com', utils.link_formatter([{'href': 'http://foo.example.com'}, {'href': 'http://bar.example.com'}]))
    self.assertEqual('http://foo.example.com (a)\nhttp://bar.example.com (b)', utils.link_formatter([{'href': 'http://foo.example.com', 'rel': 'a'}, {'href': 'http://bar.example.com', 'rel': 'b'}]))
    self.assertEqual('\n', utils.link_formatter([{'hrf': 'http://foo.example.com'}, {}]))