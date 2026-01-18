from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, setup_rax_module
def cloud_queue(module, state, name):
    for arg in (state, name):
        if not arg:
            module.fail_json(msg='%s is required for rax_queue' % arg)
    changed = False
    queues = []
    instance = {}
    cq = pyrax.queues
    if not cq:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    for queue in cq.list():
        if name != queue.name:
            continue
        queues.append(queue)
    if len(queues) > 1:
        module.fail_json(msg='Multiple Queues were matched by name')
    if state == 'present':
        if not queues:
            try:
                queue = cq.create(name)
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
        else:
            queue = queues[0]
        instance = dict(name=queue.name)
        result = dict(changed=changed, queue=instance)
        module.exit_json(**result)
    elif state == 'absent':
        if queues:
            queue = queues[0]
            try:
                queue.delete()
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
    module.exit_json(changed=changed, queue=instance)