from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def remove_subsystem_host_map_rest(self, data, type):
    if type == 'hosts':
        for item in data:
            api = 'protocols/nvme/subsystems/%s/hosts/%s' % (self.subsystem_uuid, item)
            dummy, error = rest_generic.delete_async(self.rest_api, api, None)
            if error:
                self.module.fail_json(msg='Error removing %s for subsystem %s: %s' % (item, self.parameters['subsystem'], to_native(error)), exception=traceback.format_exc())
    elif type == 'paths':
        for item in data:
            namespace_uuid = None
            for each in self.namespace_list:
                if each['name'] == item:
                    namespace_uuid = each['uuid']
            api = 'protocols/nvme/subsystem-maps/%s/%s' % (self.subsystem_uuid, namespace_uuid)
            body = {'subsystem.name': self.parameters['subsystem'], 'svm.name': self.parameters['vserver'], 'namespace.name': item}
            dummy, error = rest_generic.delete_async(self.rest_api, api, None, body=body)
            if error:
                self.module.fail_json(msg='Error removing %s for subsystem %s: %s' % (item, self.parameters['subsystem'], to_native(error)), exception=traceback.format_exc())