from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def check_passphrase_rest(self, passphrase):
    """ API does not return the passphrase
            In order to check for idempotency, check if the desired passphrase is already active"""
    params = {'onboard': {'existing_passphrase': passphrase, 'passphrase': passphrase}}
    error = self.modify_key_manager_rest(params, return_error=True)
    if not error:
        return ('unexpected_success in check_passphrase_rest', error)
    if 'Cluster-wide passphrase is incorrect.' in error:
        return ('incorrect_passphrase', error)
    if 'New passphrase cannot be same as the old passphrase.' in error:
        return ('current_passphrase', error)
    self.module.warn('Unexpected response in check_passphrase_rest: %s' % error)
    return ('unexpected_error in check_passphrase_rest', error)