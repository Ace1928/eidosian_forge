import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_get_template_url(self):
    tmpl_file = '/opt/stack/template.yaml'
    tmpl_url = 'file:///opt/stack/template.yaml'
    self.assertEqual(utils.get_template_url(tmpl_file, None), tmpl_url)
    self.assertEqual(utils.get_template_url(None, tmpl_url), tmpl_url)
    self.assertIsNone(utils.get_template_url(None, None))