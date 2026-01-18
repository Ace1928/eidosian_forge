from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
def global_config(self, cmd):
    if 'global' not in self._rendered_configuration:
        self._rendered_configuration['global'] = list()
    self._rendered_configuration['global'].extend(to_list(cmd))