from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def find_exact_match(self, records, name):
    """ with vserver, we expect an exact match
            but ONTAP transforms cluster interface names by prepending the home_port
        """
    if 'vserver' in self.parameters:
        if len(records) > 1:
            self.module.fail_json(msg='Error: unexpected records for name: %s, vserver: %s - %s' % (name, self.parameters['vserver'], records))
        return records[0] if records else None
    record = self.find_interface_record(records, None, name)
    if 'home_node' in self.parameters and self.parameters['home_node'] != 'localhost':
        home_record = self.find_interface_record(records, self.parameters['home_node'], name)
        if record and home_record:
            self.module.warn('Found both %s, selecting %s' % ([record['name'] for record in (record, home_record)], home_record['name']))
    else:
        home_node_records = []
        for home_node in self.get_cluster_node_names_rest():
            home_record = self.find_interface_record(records, home_node, name)
            if home_record:
                home_node_records.append(home_record)
        if len(home_node_records) > 1:
            self.module.fail_json(msg='Error: multiple matches for name: %s: %s.  Set home_node parameter.' % (name, [record['name'] for record in home_node_records]))
        home_record = home_node_records[0] if home_node_records else None
        if record and home_node_records:
            self.module.fail_json(msg='Error: multiple matches for name: %s: %s.  Set home_node parameter.' % (name, [record['name'] for record in (record, home_record)]))
    if home_record:
        record = home_record
    if record and name == self.parameters['interface_name'] and (name != record['name']):
        self.parameters['interface_name'] = record['name']
        self.module.warn('adjusting name from %s to %s' % (name, record['name']))
    return record