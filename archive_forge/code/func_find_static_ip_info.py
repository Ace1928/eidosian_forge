from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_static_ip_info(module, client, static_ip_name, fail_if_not_found=False):
    try:
        res = client.get_static_ip(staticIpName=static_ip_name)
    except is_boto3_error_code('NotFoundException') as e:
        if fail_if_not_found:
            module.fail_json_aws(e)
        return None
    except botocore.exceptions.ClientError as e:
        module.fail_json_aws(e)
    return res['staticIp']