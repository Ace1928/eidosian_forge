from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def replicate_source_hosts(self, hosts_data):
    self.log('Entering function replicate_source_hosts()')
    merged_result = []
    hosts_wwpn = {}
    hosts_iscsi = {}
    host_list = []
    if self.module.check_mode:
        self.changed = True
        return
    self.log('creating vdiskhostmaps on target system')
    if isinstance(hosts_data, list):
        for d in hosts_data:
            merged_result.append(d)
    elif hosts_data:
        merged_result = [hosts_data]
    for host in merged_result:
        host_list.append(host['host_name'])
    for host in host_list:
        host_wwpn_list = []
        host_iscsi_list = []
        self.log('for host %s', host)
        data = self.restapi.svc_obj_info(cmd='lshost', cmdopts=None, cmdargs=[host])
        nodes_data = data['nodes']
        for node in nodes_data:
            if 'WWPN' in node.keys():
                host_wwpn_list.append(node['WWPN'])
                hosts_wwpn[host] = host_wwpn_list
            elif 'iscsi_name' in node.keys():
                host_iscsi_list.append(node['iscsi_name'])
                hosts_iscsi[host] = host_iscsi_list
    if hosts_wwpn or hosts_iscsi:
        self.create_remote_hosts(hosts_wwpn, hosts_iscsi)