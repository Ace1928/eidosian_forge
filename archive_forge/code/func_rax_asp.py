from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import (UUID, rax_argument_spec, rax_required_together, rax_to_dict,
def rax_asp(module, at=None, change=0, cron=None, cooldown=300, desired_capacity=0, is_percent=False, name=None, policy_type=None, scaling_group=None, state='present'):
    changed = False
    au = pyrax.autoscale
    if not au:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    try:
        UUID(scaling_group)
    except ValueError:
        try:
            sg = au.find(name=scaling_group)
        except Exception as e:
            module.fail_json(msg='%s' % e.message)
    else:
        try:
            sg = au.get(scaling_group)
        except Exception as e:
            module.fail_json(msg='%s' % e.message)
    if state == 'present':
        policies = filter(lambda p: name == p.name, sg.list_policies())
        if len(policies) > 1:
            module.fail_json(msg='No unique policy match found by name')
        if at:
            args = dict(at=at)
        elif cron:
            args = dict(cron=cron)
        else:
            args = None
        if not policies:
            try:
                policy = sg.add_policy(name, policy_type=policy_type, cooldown=cooldown, change=change, is_percent=is_percent, desired_capacity=desired_capacity, args=args)
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
        else:
            policy = policies[0]
            kwargs = {}
            if policy_type != policy.type:
                kwargs['policy_type'] = policy_type
            if cooldown != policy.cooldown:
                kwargs['cooldown'] = cooldown
            if hasattr(policy, 'change') and change != policy.change:
                kwargs['change'] = change
            if hasattr(policy, 'changePercent') and is_percent is False:
                kwargs['change'] = change
                kwargs['is_percent'] = False
            elif hasattr(policy, 'change') and is_percent is True:
                kwargs['change'] = change
                kwargs['is_percent'] = True
            if hasattr(policy, 'desiredCapacity') and change:
                kwargs['change'] = change
            elif (hasattr(policy, 'change') or hasattr(policy, 'changePercent')) and desired_capacity:
                kwargs['desired_capacity'] = desired_capacity
            if hasattr(policy, 'args') and args != policy.args:
                kwargs['args'] = args
            if kwargs:
                policy.update(**kwargs)
                changed = True
        policy.get()
        module.exit_json(changed=changed, autoscale_policy=rax_to_dict(policy))
    else:
        try:
            policies = filter(lambda p: name == p.name, sg.list_policies())
            if len(policies) > 1:
                module.fail_json(msg='No unique policy match found by name')
            elif not policies:
                policy = {}
            else:
                policy.delete()
                changed = True
        except Exception as e:
            module.fail_json(msg='%s' % e.message)
        module.exit_json(changed=changed, autoscale_policy=rax_to_dict(policy))