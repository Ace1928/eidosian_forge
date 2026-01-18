from __future__ import absolute_import, division, print_function
import yaml
from ansible.module_utils.basic import missing_required_lib
from ansible.plugins.inventory import (AnsibleError, BaseInventoryPlugin,
from jinja2 import Template
from ..module_utils.cloudstack import HAS_LIB_CS
def init_cs(self):
    api_config = {'endpoint': self.get_option('api_url'), 'key': self.get_option('api_key'), 'secret': self.get_option('api_secret'), 'timeout': self.get_option('api_timeout'), 'method': self.get_option('api_http_method'), 'verify': self.get_option('api_verify_ssl_cert')}
    self._cs = CloudStack(**api_config)