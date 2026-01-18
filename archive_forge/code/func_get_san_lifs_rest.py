from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_san_lifs_rest(self, san_lifs):
    missing_lifs = []
    desired_lifs = {}
    record, record2, error, error2 = (None, None, None, None)
    for lif in san_lifs:
        if self.parameters.get('portset_type') in [None, 'mixed', 'iscsi']:
            record, error = self.get_san_lif_type(lif, 'ip')
        if self.parameters.get('portset_type') in [None, 'mixed', 'fcp']:
            record2, error2 = self.get_san_lif_type(lif, 'fc')
        if error is None and error2 is not None and record:
            error2 = None
        if error2 is None and error is not None and record2:
            error = None
        if error or error2:
            errors = [to_native(err) for err in (error, error2) if err]
            self.module.fail_json(msg='Error fetching lifs details for %s: %s' % (lif, ' - '.join(errors)), exception=traceback.format_exc())
        if record:
            desired_lifs[lif] = {'lif_type': 'ip', 'uuid': record['uuid']}
        if record2:
            desired_lifs[lif] = {'lif_type': 'fc', 'uuid': record2['uuid']}
        if record is None and record2 is None:
            missing_lifs.append(lif)
    if missing_lifs and self.parameters['state'] == 'present':
        error_msg = 'Error: lifs: %s of type %s not found in vserver %s' % (', '.join(missing_lifs), self.parameters.get('portset_type', 'fcp or iscsi'), self.parameters['vserver'])
        self.module.fail_json(msg=error_msg)
    return desired_lifs