from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def rename_server_certificate(current_cert):
    name = module.params.get('name')
    new_name = module.params.get('new_name')
    new_path = module.params.get('new_path')
    changes = dict()
    if not current_cert:
        current_cert = get_server_certificate(new_name)
    elif new_name:
        changes['NewServerCertificateName'] = new_name
    cert_metadata = current_cert.get('server_certificate_metadata', {})
    if not current_cert:
        module.fail_json(msg=f'Unable to find certificate {name}')
    current_path = cert_metadata.get('path', None)
    if new_path and current_path != new_path:
        changes['NewPath'] = new_path
    if not changes:
        return False
    if module.check_mode:
        return True
    try:
        client.update_server_certificate(aws_retry=True, ServerCertificateName=name, **changes)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f'Failed to update server certificate {name}', changes=changes)
    return True