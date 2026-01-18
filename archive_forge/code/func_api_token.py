from __future__ import (absolute_import, division, print_function)
import os
from collections import defaultdict
from json import loads
from ansible.errors import AnsibleError
from ansible.module_utils.urls import open_url
from ansible.inventory.group import to_safe_group_name
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
@property
def api_token(self):
    return self.get_option('api_token') or os.environ.get('CLOUDSCALE_API_TOKEN')