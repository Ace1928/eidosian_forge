from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
def check_partnership(self):
    if self.replication_partner_clusterid:
        merged_result = {}
        result = self.restapi.svc_obj_info(cmd='lspartnership', cmdopts=None, cmdargs=['-gui', self.replication_partner_clusterid])
        if isinstance(result, list):
            for res in result:
                merged_result = res
        else:
            merged_result = result
        if merged_result:
            self.partnership_index = merged_result.get('partnership_index')
        else:
            self.module.fail_json(msg='Partnership does not exist for the given cluster ({0}).'.format(self.replication_partner_clusterid))