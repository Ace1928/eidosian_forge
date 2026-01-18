from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.wafv2 import describe_wafv2_tags
def get_ip_set(wafv2, name, scope, id, fail_json_aws):
    try:
        response = wafv2.get_ip_set(Name=name, Scope=scope, Id=id)
    except (BotoCoreError, ClientError) as e:
        fail_json_aws(e, msg='Failed to get wafv2 ip set')
    return response