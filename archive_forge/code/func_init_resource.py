from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
def init_resource(self):
    return {'state': 'absent', self.resource_key_uuid: self._module.params.get(self.resource_key_uuid) or self._resource_data.get(self.resource_key_uuid), self.resource_key_name: self._module.params.get(self.resource_key_name) or self._resource_data.get(self.resource_key_name)}