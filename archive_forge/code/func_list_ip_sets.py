from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.wafv2 import describe_wafv2_tags
def list_ip_sets(wafv2, scope, fail_json_aws, Nextmarker=None):
    req_obj = {'Scope': scope, 'Limit': 100}
    if Nextmarker:
        req_obj['NextMarker'] = Nextmarker
    try:
        response = wafv2.list_ip_sets(**req_obj)
        if response.get('NextMarker'):
            response['IPSets'] += list_ip_sets(wafv2, scope, fail_json_aws, Nextmarker=response.get('NextMarker')).get('IPSets')
    except (BotoCoreError, ClientError) as e:
        fail_json_aws(e, msg='Failed to list wafv2 ip set')
    return response