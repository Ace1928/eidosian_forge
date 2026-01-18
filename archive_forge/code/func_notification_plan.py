from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, setup_rax_module
def notification_plan(module, state, label, critical_state, warning_state, ok_state):
    if len(label) < 1 or len(label) > 255:
        module.fail_json(msg='label must be between 1 and 255 characters long')
    changed = False
    notification_plan = None
    cm = pyrax.cloud_monitoring
    if not cm:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    existing = []
    for n in cm.list_notification_plans():
        if n.label == label:
            existing.append(n)
    if existing:
        notification_plan = existing[0]
    if state == 'present':
        should_create = False
        should_delete = False
        if len(existing) > 1:
            module.fail_json(msg='%s notification plans are labelled %s.' % (len(existing), label))
        if notification_plan:
            should_delete = critical_state and critical_state != notification_plan.critical_state or (warning_state and warning_state != notification_plan.warning_state) or (ok_state and ok_state != notification_plan.ok_state)
            if should_delete:
                notification_plan.delete()
                should_create = True
        else:
            should_create = True
        if should_create:
            notification_plan = cm.create_notification_plan(label=label, critical_state=critical_state, warning_state=warning_state, ok_state=ok_state)
            changed = True
    else:
        for np in existing:
            np.delete()
            changed = True
    if notification_plan:
        notification_plan_dict = {'id': notification_plan.id, 'critical_state': notification_plan.critical_state, 'warning_state': notification_plan.warning_state, 'ok_state': notification_plan.ok_state, 'metadata': notification_plan.metadata}
        module.exit_json(changed=changed, notification_plan=notification_plan_dict)
    else:
        module.exit_json(changed=changed)