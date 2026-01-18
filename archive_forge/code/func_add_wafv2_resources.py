from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.wafv2 import wafv2_list_web_acls
def add_wafv2_resources(wafv2, waf_arn, arn, fail_json_aws):
    try:
        response = wafv2.associate_web_acl(WebACLArn=waf_arn, ResourceArn=arn)
    except (BotoCoreError, ClientError) as e:
        fail_json_aws(e, msg='Failed to add wafv2 web acl.')
    return response