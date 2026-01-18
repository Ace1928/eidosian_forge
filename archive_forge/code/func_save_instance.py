from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, rax_to_dict, setup_rax_module
def save_instance(module, name, flavor, volume, cdb_type, cdb_version, wait, wait_timeout):
    for arg, value in dict(name=name, flavor=flavor, volume=volume, type=cdb_type, version=cdb_version).items():
        if not value:
            module.fail_json(msg='%s is required for the "rax_cdb" module' % arg)
    if not (volume >= 1 and volume <= 150):
        module.fail_json(msg='volume is required to be between 1 and 150')
    cdb = pyrax.cloud_databases
    flavors = []
    for item in cdb.list_flavors():
        flavors.append(item.id)
    if not flavor in flavors:
        module.fail_json(msg='unexisting flavor reference "%s"' % str(flavor))
    changed = False
    instance = find_instance(name)
    if not instance:
        action = 'create'
        try:
            instance = cdb.create(name=name, flavor=flavor, volume=volume, type=cdb_type, version=cdb_version)
        except Exception as e:
            module.fail_json(msg='%s' % e.message)
        else:
            changed = True
    else:
        action = None
        if instance.volume.size != volume:
            action = 'resize'
            if instance.volume.size > volume:
                module.fail_json(changed=False, action=action, msg='The new volume size must be larger than the current volume size', cdb=rax_to_dict(instance))
            instance.resize_volume(volume)
            changed = True
        if int(instance.flavor.id) != flavor:
            action = 'resize'
            pyrax.utils.wait_until(instance, 'status', 'ACTIVE', attempts=wait_timeout)
            instance.resize(flavor)
            changed = True
    if wait:
        pyrax.utils.wait_until(instance, 'status', 'ACTIVE', attempts=wait_timeout)
    if wait and instance.status != 'ACTIVE':
        module.fail_json(changed=changed, action=action, cdb=rax_to_dict(instance), msg='Timeout waiting for "%s" databases instance to be created' % name)
    module.exit_json(changed=changed, action=action, cdb=rax_to_dict(instance))