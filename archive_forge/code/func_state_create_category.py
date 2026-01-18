from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.community.vmware.plugins.module_utils.vmware import connect_to_api
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def state_create_category(self):
    """Create category."""
    category_spec = self.category_service.CreateSpec()
    category_spec.name = self.category_name
    category_spec.description = self.params.get('category_description')
    if self.params.get('category_cardinality') == 'single':
        category_spec.cardinality = CategoryModel.Cardinality.SINGLE
    else:
        category_spec.cardinality = CategoryModel.Cardinality.MULTIPLE
    associable_object_types = self.params.get('associable_object_types')

    def append_namespace(object_name):
        return '%s:%s' % (XMLNS_VMODL_BASE, object_name)
    associable_data = {'cluster': append_namespace('ClusterComputeResource'), 'datastore': append_namespace('Datastore'), 'datastore cluster': append_namespace('StoragePod'), 'folder': append_namespace('Folder'), 'host': append_namespace('HostSystem'), 'library item': append_namespace('com.vmware.content.library.Item'), 'datacenter': 'Datacenter', 'distributed port group': 'DistributedVirtualPortgroup', 'distributed switch': ['VmwareDistributedVirtualSwitch', 'DistributedVirtualSwitch'], 'content library': 'com.vmware.content.Library', 'resource pool': 'ResourcePool', 'vapp': 'VirtualApp', 'virtual machine': 'VirtualMachine', 'network': ['Network', 'HostNetwork', 'OpaqueNetwork'], 'host network': 'HostNetwork', 'opaque network': 'OpaqueNetwork'}
    obj_types_set = []
    if associable_object_types:
        for obj_type in associable_object_types:
            lower_obj_type = obj_type.lower()
            if lower_obj_type == 'all objects':
                if LooseVersion(self.content.about.version) < LooseVersion('7'):
                    break
                for category in list(associable_data.values()):
                    if isinstance(category, list):
                        obj_types_set.extend(category)
                    else:
                        obj_types_set.append(category)
                break
            if lower_obj_type in associable_data:
                value = associable_data.get(lower_obj_type)
                if isinstance(value, list):
                    obj_types_set.extend(value)
                else:
                    obj_types_set.append(value)
            else:
                obj_types_set.append(obj_type)
    category_spec.associable_types = set(obj_types_set)
    category_id = ''
    try:
        category_id = self.category_service.create(category_spec)
    except Error as error:
        self.module.fail_json(msg='%s' % self.get_error_message(error))
    msg = 'No category created'
    changed = False
    if category_id:
        changed = True
        msg = "Category '%s' created." % category_spec.name
    self.module.exit_json(changed=changed, category_results=dict(msg=msg, category_id=category_id))