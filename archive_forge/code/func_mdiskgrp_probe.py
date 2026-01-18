from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
def mdiskgrp_probe(self, data):
    props = {}
    if self.noprovisioningpolicy and data.get('provisioning_policy_name', ''):
        props['noprovisioningpolicy'] = self.noprovisioningpolicy
    if self.provisioningpolicy and self.provisioningpolicy != data.get('provisioning_policy_name', ''):
        props['provisioningpolicy'] = self.provisioningpolicy
    if self.noownershipgroup and data.get('owner_name', ''):
        props['noownershipgroup'] = self.noownershipgroup
    if self.ownershipgroup and self.ownershipgroup != data.get('owner_name', ''):
        props['ownershipgroup'] = self.ownershipgroup
    if self.vdiskprotectionenabled and self.vdiskprotectionenabled != data.get('vdisk_protectionenabled', ''):
        props['vdiskprotectionenabled'] = self.vdiskprotectionenabled
    if self.warning and self.warning != data.get('warning', ''):
        props['warning'] = str(self.warning) + '%'
    if self.replicationpoollinkuid and self.replicationpoollinkuid != data.get('replication_pool_link_uid', ''):
        props['replicationpoollinkuid'] = self.replicationpoollinkuid
    if self.resetreplicationpoollinkuid:
        props['resetreplicationpoollinkuid'] = self.resetreplicationpoollinkuid
    if self.etfcmoverallocationmax:
        if '%' not in self.etfcmoverallocationmax and self.etfcmoverallocationmax != 'off':
            self.etfcmoverallocationmax += '%'
        if self.etfcmoverallocationmax != data.get('easy_tier_fcm_over_allocation_max', ''):
            props['etfcmoverallocationmax'] = self.etfcmoverallocationmax
    if self.replication_partner_clusterid:
        self.check_partnership()
        bit_mask = '1'.ljust(int(self.partnership_index) + 1, '0')
        if bit_mask.zfill(64) != data.get('replication_pool_linked_systems_mask', ''):
            props['replicationpoollinkedsystemsmask'] = bit_mask
    self.log("mdiskgrp_probe props='%s'", props)
    return props