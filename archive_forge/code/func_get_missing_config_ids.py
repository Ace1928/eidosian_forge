from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def get_missing_config_ids(self):
    """
        Resolve missing config ids by looking them up by name
        """
    config_names = [config['config_name'] for config in self.client.module.params.get('configs') or [] if config['config_id'] is None]
    if not config_names:
        return {}
    configs = self.client.configs(filters={'name': config_names})
    configs = dict(((config['Spec']['Name'], config['ID']) for config in configs if config['Spec']['Name'] in config_names))
    for config_name in config_names:
        if config_name not in configs:
            self.client.fail('Could not find a config named "%s"' % config_name)
    return configs