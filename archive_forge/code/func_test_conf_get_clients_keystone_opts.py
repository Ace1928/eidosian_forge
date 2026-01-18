import json
from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
import requests
from heat.api.aws import ec2token
from heat.api.aws import exception
from heat.common import wsgi
from heat.tests import common
from heat.tests import utils
def test_conf_get_clients_keystone_opts(self):
    cfg.CONF.set_default('auth_uri', None, group='ec2authtoken')
    cfg.CONF.set_default('auth_uri', 'http://192.0.2.9', group='clients_keystone')
    with mock.patch('keystoneauth1.discover.Discover') as discover:

        class MockDiscover(object):

            def url_for(self, endpoint):
                return 'http://192.0.2.9/v3/'
        discover.return_value = MockDiscover()
        ec2 = ec2token.EC2Token(app=None, conf={})
        self.assertEqual('http://192.0.2.9/v3/ec2tokens', ec2._conf_get_keystone_ec2_uri('http://192.0.2.9/v3/'))