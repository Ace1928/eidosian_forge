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
def test_conf_ssl_insecure_option(self):
    ec2 = ec2token.EC2Token(app=None, conf={})
    cfg.CONF.set_default('insecure', 'True', group='ec2authtoken')
    cfg.CONF.set_default('ca_file', None, group='ec2authtoken')
    self.assertFalse(ec2.ssl_options['verify'])