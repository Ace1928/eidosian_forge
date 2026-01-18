from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import (
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible.module_utils._text import to_native
def state_register_vasa(self):
    """
        Register VASA provider with vcenter
        """
    changed, result = (True, None)
    vasa_provider_spec = sms.provider.VasaProviderSpec()
    vasa_provider_spec.name = self.vasa_name
    vasa_provider_spec.username = self.vasa_username
    vasa_provider_spec.password = self.vasa_password
    vasa_provider_spec.url = self.vasa_url
    vasa_provider_spec.certificate = self.vasa_certificate
    try:
        if not self.module.check_mode:
            task = self.storage_manager.RegisterProvider_Task(vasa_provider_spec)
            changed, result = wait_for_sms_task(task)
            if isinstance(result, sms.fault.CertificateNotTrusted):
                vasa_provider_spec.certificate = result.certificate
                task = self.storage_manager.RegisterProvider_Task(vasa_provider_spec)
                changed, result = wait_for_sms_task(task)
            if isinstance(result, sms.provider.VasaProvider):
                provider_info = result.QueryProviderInfo()
                temp_provider_info = {'name': provider_info.name, 'uid': provider_info.uid, 'description': provider_info.description, 'version': provider_info.version, 'certificate_status': provider_info.certificateStatus, 'url': provider_info.url, 'status': provider_info.status, 'related_storage_array': []}
                for a in provider_info.relatedStorageArray:
                    temp_storage_array = {'active': str(a.active), 'array_id': a.arrayId, 'manageable': str(a.manageable), 'priority': str(a.priority)}
                    temp_provider_info['related_storage_array'].append(temp_storage_array)
                result = temp_provider_info
        self.module.exit_json(changed=changed, result=result)
    except TaskError as task_err:
        self.module.fail_json(msg='Failed to register VASA provider due to task exception %s' % to_native(task_err))
    except Exception as generic_exc:
        self.module.fail_json(msg='Failed to register VASA due to generic exception %s' % to_native(generic_exc))