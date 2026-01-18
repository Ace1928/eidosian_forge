from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def disassociate_ip_and_device(ec2, module, address, device_id, check_mode, is_instance=True):
    if not address_is_associated_with_device(ec2, module, address, device_id, is_instance):
        return {'changed': False}
    if not check_mode:
        try:
            if address['Domain'] == 'vpc':
                ec2.disassociate_address(AssociationId=address['AssociationId'], aws_retry=True)
            else:
                ec2.disassociate_address(PublicIp=address['PublicIp'], aws_retry=True)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg='Dissassociation of Elastic IP failed')
    return {'changed': True}