from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def find_interface_record(self, records, home_node, name):
    full_name = '%s_%s' % (home_node, name) if home_node is not None else name
    full_name_records = [record for record in records if record['name'] == full_name]
    if len(full_name_records) > 1:
        self.module.fail_json(msg='Error: multiple records for: %s - %s' % (full_name, full_name_records))
    return full_name_records[0] if full_name_records else None