from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.elb_utils import get_elb_listener_rules
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ApplicationLoadBalancer
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListener
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListenerRule
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListenerRules
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListeners
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def create_or_update_alb(alb_obj):
    """Create ALB or modify main attributes. json_exit here"""
    if alb_obj.elb:
        if not alb_obj.compare_subnets():
            if alb_obj.module.check_mode:
                alb_obj.module.exit_json(changed=True, msg='Would have updated ALB if not in check mode.')
            alb_obj.modify_subnets()
        if not alb_obj.compare_security_groups():
            if alb_obj.module.check_mode:
                alb_obj.module.exit_json(changed=True, msg='Would have updated ALB if not in check mode.')
            alb_obj.modify_security_groups()
        if not alb_obj.compare_elb_attributes():
            if alb_obj.module.check_mode:
                alb_obj.module.exit_json(changed=True, msg='Would have updated ALB if not in check mode.')
            alb_obj.update_elb_attributes()
            alb_obj.modify_elb_attributes()
        if alb_obj.tags is not None:
            tags_need_modify, tags_to_delete = compare_aws_tags(boto3_tag_list_to_ansible_dict(alb_obj.elb['tags']), boto3_tag_list_to_ansible_dict(alb_obj.tags), alb_obj.purge_tags)
            if alb_obj.module.check_mode and (tags_need_modify or tags_to_delete):
                alb_obj.module.exit_json(changed=True, msg='Would have updated ALB if not in check mode.')
            if tags_to_delete:
                alb_obj.delete_tags(tags_to_delete)
            if tags_need_modify:
                alb_obj.modify_tags()
    else:
        if alb_obj.module.check_mode:
            alb_obj.module.exit_json(changed=True, msg='Would have created ALB if not in check mode.')
        alb_obj.create_elb()
        alb_obj.update_elb_attributes()
        alb_obj.modify_elb_attributes()
    listeners_obj = ELBListeners(alb_obj.connection, alb_obj.module, alb_obj.elb['LoadBalancerArn'])
    listeners_to_add, listeners_to_modify, listeners_to_delete = listeners_obj.compare_listeners()
    if alb_obj.module.check_mode and (listeners_to_add or listeners_to_modify or listeners_to_delete):
        alb_obj.module.exit_json(changed=True, msg='Would have updated ALB if not in check mode.')
    for listener_to_delete in listeners_to_delete:
        listener_obj = ELBListener(alb_obj.connection, alb_obj.module, listener_to_delete, alb_obj.elb['LoadBalancerArn'])
        listener_obj.delete()
        listeners_obj.changed = True
    for listener_to_add in listeners_to_add:
        listener_obj = ELBListener(alb_obj.connection, alb_obj.module, listener_to_add, alb_obj.elb['LoadBalancerArn'])
        listener_obj.add()
        listeners_obj.changed = True
    for listener_to_modify in listeners_to_modify:
        listener_obj = ELBListener(alb_obj.connection, alb_obj.module, listener_to_modify, alb_obj.elb['LoadBalancerArn'])
        listener_obj.modify()
        listeners_obj.changed = True
    if listeners_obj.changed:
        alb_obj.changed = True
    for listener in listeners_obj.listeners:
        if 'Rules' in listener:
            rules_obj = ELBListenerRules(alb_obj.connection, alb_obj.module, alb_obj.elb['LoadBalancerArn'], listener['Rules'], listener['Port'])
            rules_to_add, rules_to_modify, rules_to_delete, rules_to_set_priority = rules_obj.compare_rules()
            if alb_obj.module.check_mode and (rules_to_add or rules_to_modify or rules_to_delete or rules_to_set_priority):
                alb_obj.module.exit_json(changed=True, msg='Would have updated ALB if not in check mode.')
            if rules_to_set_priority:
                rule_obj = ELBListenerRule(alb_obj.connection, alb_obj.module, rules_to_set_priority, rules_obj.listener_arn)
                rule_obj.set_rule_priorities()
                alb_obj.changed |= rule_obj.changed
            if alb_obj.module.params['purge_rules']:
                for rule in rules_to_delete:
                    rule_obj = ELBListenerRule(alb_obj.connection, alb_obj.module, {'RuleArn': rule}, rules_obj.listener_arn)
                    rule_obj.delete()
                    alb_obj.changed = True
            for rule in rules_to_add:
                rule_obj = ELBListenerRule(alb_obj.connection, alb_obj.module, rule, rules_obj.listener_arn)
                rule_obj.create()
                alb_obj.changed = True
            for rule in rules_to_modify:
                rule_obj = ELBListenerRule(alb_obj.connection, alb_obj.module, rule, rules_obj.listener_arn)
                rule_obj.modify()
                alb_obj.changed = True
    if alb_obj.module.params.get('ip_address_type') and alb_obj.elb_ip_addr_type != alb_obj.module.params.get('ip_address_type'):
        if alb_obj.module.check_mode:
            alb_obj.module.exit_json(changed=True, msg='Would have updated ALB if not in check mode.')
        alb_obj.modify_ip_address_type(alb_obj.module.params.get('ip_address_type'))
    if alb_obj.module.check_mode:
        alb_obj.module.exit_json(changed=False, msg='IN CHECK MODE - no changes to make to ALB specified.')
    alb_obj.update()
    listeners_obj.update()
    alb_obj.update_elb_attributes()
    snaked_alb = camel_dict_to_snake_dict(alb_obj.elb)
    snaked_alb.update(camel_dict_to_snake_dict(alb_obj.elb_attributes))
    snaked_alb['listeners'] = []
    for listener in listeners_obj.current_listeners:
        listener['rules'] = get_elb_listener_rules(alb_obj.connection, alb_obj.module, listener['ListenerArn'])
        snaked_alb['listeners'].append(camel_dict_to_snake_dict(listener))
    snaked_alb['tags'] = boto3_tag_list_to_ansible_dict(snaked_alb['tags'])
    snaked_alb['ip_address_type'] = alb_obj.get_elb_ip_address_type()
    alb_obj.module.exit_json(changed=alb_obj.changed, **snaked_alb)