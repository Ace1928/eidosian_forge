from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListener
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListeners
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import NetworkLoadBalancer
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_elb(elb_obj):
    """Create ELB or modify main attributes. json_exit here"""
    if elb_obj.elb:
        if not elb_obj.compare_subnets():
            elb_obj.modify_subnets()
        if elb_obj.tags is not None:
            tags_need_modify, tags_to_delete = compare_aws_tags(boto3_tag_list_to_ansible_dict(elb_obj.elb['tags']), boto3_tag_list_to_ansible_dict(elb_obj.tags), elb_obj.purge_tags)
            if tags_to_delete:
                elb_obj.delete_tags(tags_to_delete)
            if tags_need_modify:
                elb_obj.modify_tags()
    else:
        elb_obj.create_elb()
    elb_obj.update_elb_attributes()
    elb_obj.modify_elb_attributes()
    listeners_obj = ELBListeners(elb_obj.connection, elb_obj.module, elb_obj.elb['LoadBalancerArn'])
    listeners_to_add, listeners_to_modify, listeners_to_delete = listeners_obj.compare_listeners()
    for listener_to_delete in listeners_to_delete:
        listener_obj = ELBListener(elb_obj.connection, elb_obj.module, listener_to_delete, elb_obj.elb['LoadBalancerArn'])
        listener_obj.delete()
        listeners_obj.changed = True
    for listener_to_add in listeners_to_add:
        listener_obj = ELBListener(elb_obj.connection, elb_obj.module, listener_to_add, elb_obj.elb['LoadBalancerArn'])
        listener_obj.add()
        listeners_obj.changed = True
    for listener_to_modify in listeners_to_modify:
        listener_obj = ELBListener(elb_obj.connection, elb_obj.module, listener_to_modify, elb_obj.elb['LoadBalancerArn'])
        listener_obj.modify()
        listeners_obj.changed = True
    if listeners_obj.changed:
        elb_obj.changed = True
    if elb_obj.module.params.get('ip_address_type') is not None:
        elb_obj.modify_ip_address_type(elb_obj.module.params.get('ip_address_type'))
    elb_obj.update()
    listeners_obj.update()
    elb_obj.update_elb_attributes()
    snaked_elb = camel_dict_to_snake_dict(elb_obj.elb)
    snaked_elb.update(camel_dict_to_snake_dict(elb_obj.elb_attributes))
    snaked_elb['listeners'] = []
    for listener in listeners_obj.current_listeners:
        snaked_elb['listeners'].append(camel_dict_to_snake_dict(listener))
    snaked_elb['tags'] = boto3_tag_list_to_ansible_dict(snaked_elb['tags'])
    snaked_elb['ip_address_type'] = elb_obj.get_elb_ip_address_type()
    elb_obj.module.exit_json(changed=elb_obj.changed, load_balancer=snaked_elb, **snaked_elb)