from __future__ import absolute_import, division, print_function
import json
import traceback
import re
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.six import PY2
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def diff_template(self, template_json_a, template_json_b):
    template_json_a = self.filter_template(template_json_a)
    template_json_b = self.filter_template(template_json_b)
    if self.ordered_json(template_json_a) == self.ordered_json(template_json_b):
        return False
    return True