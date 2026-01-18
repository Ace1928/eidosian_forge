from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import (NON_CALLABLES,
def cloud_block_storage_attachments(module, state, volume, server, device, wait, wait_timeout):
    cbs = pyrax.cloud_blockstorage
    cs = pyrax.cloudservers
    if cbs is None or cs is None:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    changed = False
    instance = {}
    volume = rax_find_volume(module, pyrax, volume)
    if not volume:
        module.fail_json(msg='No matching storage volumes were found')
    if state == 'present':
        server = rax_find_server(module, pyrax, server)
        if volume.attachments and volume.attachments[0]['server_id'] == server.id:
            changed = False
        elif volume.attachments:
            module.fail_json(msg='Volume is attached to another server')
        else:
            try:
                volume.attach_to_instance(server, mountpoint=device)
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
            volume.get()
        for key, value in vars(volume).items():
            if isinstance(value, NON_CALLABLES) and (not key.startswith('_')):
                instance[key] = value
        result = dict(changed=changed)
        if volume.status == 'error':
            result['msg'] = '%s failed to build' % volume.id
        elif wait:
            attempts = wait_timeout // 5
            pyrax.utils.wait_until(volume, 'status', 'in-use', interval=5, attempts=attempts)
        volume.get()
        result['volume'] = rax_to_dict(volume)
        if 'msg' in result:
            module.fail_json(**result)
        else:
            module.exit_json(**result)
    elif state == 'absent':
        server = rax_find_server(module, pyrax, server)
        if volume.attachments and volume.attachments[0]['server_id'] == server.id:
            try:
                volume.detach()
                if wait:
                    pyrax.utils.wait_until(volume, 'status', 'available', interval=3, attempts=0, verbose=False)
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
            volume.get()
            changed = True
        elif volume.attachments:
            module.fail_json(msg='Volume is attached to another server')
        result = dict(changed=changed, volume=rax_to_dict(volume))
        if volume.status == 'error':
            result['msg'] = '%s failed to build' % volume.id
        if 'msg' in result:
            module.fail_json(**result)
        else:
            module.exit_json(**result)
    module.exit_json(changed=changed, volume=instance)