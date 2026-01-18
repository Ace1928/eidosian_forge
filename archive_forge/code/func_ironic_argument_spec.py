from ansible.module_utils.basic import AnsibleModule
from ansible_collections.openstack.cloud.plugins.module_utils.openstack import openstack_full_argument_spec
def ironic_argument_spec(**kwargs):
    spec = dict(auth_type=dict(), ironic_url=dict())
    spec.update(kwargs)
    return openstack_full_argument_spec(**spec)