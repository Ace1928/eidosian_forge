from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_nvme(self):
    """
        Create NVMe service
        """
    if self.use_rest:
        return self.create_nvme_rest()
    nvme_create = netapp_utils.zapi.NaElement('nvme-create')
    if self.parameters.get('status_admin') is not None:
        options = {'is-available': self.na_helper.get_value_for_bool(False, self.parameters['status_admin'])}
        nvme_create.translate_struct(options)
    try:
        self.server.invoke_successfully(nvme_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating nvme for vserver %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())