from __future__ import absolute_import, division, print_function
import traceback
import json
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.six import PY2
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def load_yaml_template(self, template_yaml, omit_date=False):
    if omit_date:
        yaml_lines = template_yaml.splitlines(True)
        for index, line in enumerate(yaml_lines):
            if 'date:' in line:
                del yaml_lines[index]
                return ''.join(yaml_lines)
    else:
        return template_yaml