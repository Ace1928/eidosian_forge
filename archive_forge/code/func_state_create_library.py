from __future__ import absolute_import, division, print_function
import uuid
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
def state_create_library(self):
    if not self.datastore_name:
        self.module.fail_json(msg='datastore_name must be specified for create operations')
    datastore_id = self.pyv.find_datastore_by_name(datastore_name=self.datastore_name)
    if not datastore_id:
        self.module.fail_json(msg='Failed to find the datastore %s' % self.datastore_name)
    self.datastore_id = datastore_id._moId
    storage_backings = []
    storage_backing = StorageBacking(type=StorageBacking.Type.DATASTORE, datastore_id=self.datastore_id)
    storage_backings.append(storage_backing)
    create_spec = LibraryModel()
    create_spec.name = self.library_name
    create_spec.description = self.library_description
    self.library_types = {'local': create_spec.LibraryType.LOCAL, 'subscribed': create_spec.LibraryType.SUBSCRIBED}
    create_spec.type = self.library_types[self.library_type]
    create_spec.storage_backings = storage_backings
    if self.library_type == 'subscribed':
        subscription_info = self.set_subscription_spec()
        subscription_info.authentication_method = SubscriptionInfo.AuthenticationMethod.NONE
        create_spec.subscription_info = subscription_info
    self.create_update(spec=create_spec)