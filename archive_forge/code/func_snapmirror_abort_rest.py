from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def snapmirror_abort_rest(self):
    """
        Abort a SnapMirror relationship in progress using REST
        """
    uuid = self.get_relationship_uuid(after_create=False)
    transfer_uuid = self.parameters.get('transfer_uuid')
    if uuid is None or transfer_uuid is None:
        self.module.fail_json(msg='Error in aborting SnapMirror: unable to get either uuid: %s or transfer_uuid: %s.' % (uuid, transfer_uuid))
    api = 'snapmirror/relationships/%s/transfers' % uuid
    body = {'state': 'aborted'}
    dummy, error = rest_generic.patch_async(self.rest_api, api, transfer_uuid, body)
    if error:
        self.module.fail_json(msg='Error aborting SnapMirror: %s' % to_native(error), exception=traceback.format_exc())