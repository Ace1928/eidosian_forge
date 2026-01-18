from __future__ import absolute_import, division, print_function
import traceback
import json
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.six import PY2
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def dump_template(self, template_id, template_type='json', omit_date=False):
    try:
        dump = self._zapi.configuration.export({'format': template_type, 'options': {'templates': template_id}})
        if template_type == 'xml':
            xmlroot = ET.fromstring(dump.encode('utf-8'))
            if omit_date:
                date = xmlroot.find('.date')
                if date is not None:
                    xmlroot.remove(date)
            if PY2:
                return str(ET.tostring(xmlroot, encoding='utf-8'))
            else:
                return str(ET.tostring(xmlroot, encoding='utf-8').decode('utf-8'))
        elif template_type == 'yaml':
            return self.load_yaml_template(dump, omit_date)
        else:
            return self.load_json_template(dump, omit_date)
    except Exception as e:
        self._module.fail_json(msg='Unable to export template: %s' % e)