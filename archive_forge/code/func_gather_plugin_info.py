from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def gather_plugin_info(self):
    result = dict(changed=False, extension_info=[])
    ext_manager = self.content.extensionManager
    if not ext_manager:
        self.module.exit_json(**result)
    for ext in ext_manager.extensionList:
        ext_info = dict(extension_label=ext.description.label, extension_summary=ext.description.summary, extension_key=ext.key, extension_company=ext.company, extension_version=ext.version, extension_type=ext.type if ext.type else '', extension_subject_name=ext.subjectName if ext.subjectName else '', extension_last_heartbeat_time=ext.lastHeartbeatTime)
        result['extension_info'].append(ext_info)
    self.module.exit_json(**result)