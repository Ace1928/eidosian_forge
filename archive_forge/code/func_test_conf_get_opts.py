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
def test_conf_get_opts(self):
    cfg.CONF.set_default('auth_uri', 'http://192.0.2.9/v2.0/', group='ec2authtoken')
    cfg.CONF.set_default('auth_uri', 'this-should-be-ignored', group='clients_keystone')
    ec2 = ec2token.EC2Token(app=None, conf={})
    self.assertEqual('http://192.0.2.9/v2.0/', ec2._conf_get('auth_uri'))
    self.assertEqual('http://192.0.2.9/v2.0/ec2tokens', ec2._conf_get_keystone_ec2_uri('http://192.0.2.9/v2.0/'))