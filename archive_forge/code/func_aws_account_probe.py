from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def aws_account_probe(self):
    updates = {}
    if self.encrypt and self.encrypt != self.aws_data.get('encrypt', ''):
        self.module.fail_json(msg='Parameter not supported for update operation: encrypt')
    if self.bucketprefix and self.bucketprefix != self.aws_data.get('awss3_bucket_prefix', ''):
        self.module.fail_json(msg='Parameter not supported for update operation: bucketprefix')
    if self.region and self.region != self.aws_data.get('awss3_region', ''):
        self.module.fail_json(msg='Parameter not supported for update operation: region')
    self.rename_validation(updates)
    params = [('upbandwidthmbits', self.aws_data.get('up_bandwidth_mbits')), ('downbandwidthmbits', self.aws_data.get('down_bandwidth_mbits')), ('mode', self.aws_data.get('mode')), ('importsystem', self.aws_data.get('import_system_name'))]
    for k, v in params:
        if getattr(self, k) and getattr(self, k) != v:
            updates[k] = getattr(self, k)
    if self.accesskeyid and self.aws_data.get('awss3_access_key_id') != self.accesskeyid:
        updates['accesskeyid'] = self.accesskeyid
        updates['secretaccesskey'] = self.secretaccesskey
        if self.ignorefailures:
            updates['ignorefailures'] = self.ignorefailures
    if self.refresh and self.aws_data.get('refreshing') == 'no':
        updates['refresh'] = self.refresh
    if self.resetusagehistory:
        updates['resetusagehistory'] = self.resetusagehistory
    return updates