from __future__ import absolute_import, division, print_function
import traceback
import json
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.six import PY2
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def load_json_template(self, template_json, omit_date=False):
    try:
        jsondoc = json.loads(template_json)
        if omit_date and 'date' in jsondoc['zabbix_export']:
            del jsondoc['zabbix_export']['date']
        return jsondoc
    except ValueError as e:
        self._module.fail_json(msg='Invalid JSON provided', details=to_native(e), exception=traceback.format_exc())