from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import (VOLUME_STATUS, rax_argument_spec, rax_find_image, rax_find_volume,
def cloud_block_storage(module, state, name, description, meta, size, snapshot_id, volume_type, wait, wait_timeout, image):
    changed = False
    volume = None
    instance = {}
    cbs = pyrax.cloud_blockstorage
    if cbs is None:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    if image:
        if LooseVersion(pyrax.version.version) < LooseVersion('1.9.3'):
            module.fail_json(msg='Creating a bootable volume requires pyrax>=1.9.3')
        image = rax_find_image(module, pyrax, image)
    volume = rax_find_volume(module, pyrax, name)
    if state == 'present':
        if not volume:
            kwargs = dict()
            if image:
                kwargs['image'] = image
            try:
                volume = cbs.create(name, size=size, volume_type=volume_type, description=description, metadata=meta, snapshot_id=snapshot_id, **kwargs)
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
            else:
                if wait:
                    attempts = wait_timeout // 5
                    pyrax.utils.wait_for_build(volume, interval=5, attempts=attempts)
        volume.get()
        instance = rax_to_dict(volume)
        result = dict(changed=changed, volume=instance)
        if volume.status == 'error':
            result['msg'] = '%s failed to build' % volume.id
        elif wait and volume.status not in VOLUME_STATUS:
            result['msg'] = 'Timeout waiting on %s' % volume.id
        if 'msg' in result:
            module.fail_json(**result)
        else:
            module.exit_json(**result)
    elif state == 'absent':
        if volume:
            instance = rax_to_dict(volume)
            try:
                volume.delete()
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
    module.exit_json(changed=changed, volume=instance)