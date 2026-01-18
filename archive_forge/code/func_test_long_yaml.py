import os
from unittest import mock
import re
import yaml
from heat.common import config
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.tests import common
from heat.tests import utils
def test_long_yaml(self):
    template = {'HeatTemplateFormatVersion': '2012-12-12'}
    config.cfg.CONF.set_override('max_template_size', 10)
    template['Resources'] = ['a'] * int(config.cfg.CONF.max_template_size / 3)
    limit = config.cfg.CONF.max_template_size
    long_yaml = yaml.safe_dump(template)
    self.assertGreater(len(long_yaml), limit)
    ex = self.assertRaises(exception.RequestLimitExceeded, template_format.parse, long_yaml)
    msg = 'Request limit exceeded: Template size (%(actual_len)s bytes) exceeds maximum allowed size (%(limit)s bytes).' % {'actual_len': len(str(long_yaml)), 'limit': config.cfg.CONF.max_template_size}
    self.assertEqual(msg, str(ex))