from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def host_probe(self, data):
    props = []
    if self.hostcluster and self.nohostcluster:
        self.module.fail_json(msg='You must not pass in both hostcluster and nohostcluster to the module.')
    if self.hostcluster and self.hostcluster != data['host_cluster_name']:
        if data['host_cluster_name'] != '':
            self.module.fail_json(msg='Host already belongs to hostcluster [%s]' % data['host_cluster_name'])
        else:
            props += ['hostcluster']
    if self.type:
        if self.type != data['type']:
            props += ['type']
    if self.fcwwpn:
        self.existing_fcwwpn = [node['WWPN'] for node in data['nodes'] if 'WWPN' in node]
        self.input_fcwwpn = self.fcwwpn.upper().split(':')
        if set(self.existing_fcwwpn).symmetric_difference(set(self.input_fcwwpn)):
            props += ['fcwwpn']
    if self.iscsiname:
        self.existing_iscsiname = [node['iscsi_name'] for node in data['nodes'] if 'iscsi_name' in node]
        self.input_iscsiname = self.iscsiname.split(',')
        if set(self.existing_iscsiname).symmetric_difference(set(self.input_iscsiname)):
            props += ['iscsiname']
    if self.nqn:
        self.existing_nqn = [node['nqn'] for node in data['nodes'] if 'nqn' in node]
        self.input_nqn = self.nqn.split(',')
        if set(self.existing_nqn).symmetric_difference(set(self.input_nqn)):
            props += ['nqn']
    if self.site:
        if self.site != data['site_name']:
            props += ['site']
    if self.nohostcluster:
        if data['host_cluster_name'] != '':
            props += ['nohostcluster']
    if self.portset:
        if self.portset != data['portset_name']:
            props += ['portset']
    self.log("host_probe props='%s'", props)
    return props