from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec, cs_required_together
def get_storage_providers(self, storage_type='image'):
    args = {'type': storage_type}
    storage_provides = self.query_api('listStorageProviders', **args)
    return [provider.get('name') for provider in storage_provides.get('dataStoreProvider')]