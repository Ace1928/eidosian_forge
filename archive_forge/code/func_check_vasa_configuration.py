from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import (
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible.module_utils._text import to_native
def check_vasa_configuration(self):
    """
        Check VASA configuration
        Returns: 'Present' if VASA provider exists, else 'absent'

        """
    self.vasa_provider_info = None
    self.storage_manager = self.sms_si.QueryStorageManager()
    storage_providers = self.storage_manager.QueryProvider()
    try:
        for provider in storage_providers:
            provider_info = provider.QueryProviderInfo()
            if provider_info.name == self.vasa_name:
                if provider_info.url != self.vasa_url:
                    raise Exception("VASA provider '%s' URL '%s' is inconsistent  with task parameter '%s'" % (self.vasa_name, provider_info.url, self.vasa_url))
                self.vasa_provider_info = provider_info
                break
        if self.vasa_provider_info is None:
            return 'absent'
        return 'present'
    except Exception as generic_exc:
        self.module.fail_json(msg='Failed to check configuration due to generic exception %s' % to_native(generic_exc))