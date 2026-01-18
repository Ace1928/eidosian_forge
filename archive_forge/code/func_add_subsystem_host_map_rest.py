from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_subsystem_host_map_rest(self, data, type):
    if type == 'hosts':
        records = [{'nqn': item} for item in data]
        api = 'protocols/nvme/subsystems/%s/hosts' % self.subsystem_uuid
        body = {'records': records}
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error adding %s for subsystem %s: %s' % (records, self.parameters['subsystem'], to_native(error)), exception=traceback.format_exc())
    elif type == 'paths':
        api = 'protocols/nvme/subsystem-maps'
        for item in data:
            body = {'subsystem.name': self.parameters['subsystem'], 'svm.name': self.parameters['vserver'], 'namespace.name': item}
            dummy, error = rest_generic.post_async(self.rest_api, api, body)
            if error:
                self.module.fail_json(msg='Error adding %s for subsystem %s: %s' % (item, self.parameters['subsystem'], to_native(error)), exception=traceback.format_exc())