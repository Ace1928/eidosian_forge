from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def hosted_zone_details():
    hosted_zone_invocations = {'details': get_hosted_zone, 'list': list_hosted_zones, 'list_by_name': list_hosted_zones_by_name, 'count': get_count, 'tags': get_resource_tags}
    results = hosted_zone_invocations[module.params.get('hosted_zone_method')]()
    return results