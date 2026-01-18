from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_static_ip(module, client, static_ip_name):
    inst = find_static_ip_info(module, client, static_ip_name)
    if inst:
        module.exit_json(changed=False, static_ip=camel_dict_to_snake_dict(inst))
    else:
        create_params = {'staticIpName': static_ip_name}
        try:
            client.allocate_static_ip(**create_params)
        except botocore.exceptions.ClientError as e:
            module.fail_json_aws(e)
        inst = find_static_ip_info(module, client, static_ip_name, fail_if_not_found=True)
        module.exit_json(changed=True, static_ip=camel_dict_to_snake_dict(inst))