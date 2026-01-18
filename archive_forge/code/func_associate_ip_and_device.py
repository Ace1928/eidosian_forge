from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def associate_ip_and_device(ec2, module, address, private_ip_address, device_id, allow_reassociation, check_mode, is_instance=True):
    if address_is_associated_with_device(ec2, module, address, device_id, is_instance):
        return {'changed': False}
    if not check_mode:
        if is_instance:
            try:
                params = dict(InstanceId=device_id, AllowReassociation=allow_reassociation)
                if private_ip_address:
                    params['PrivateIpAddress'] = private_ip_address
                if address['Domain'] == 'vpc':
                    params['AllocationId'] = address['AllocationId']
                else:
                    params['PublicIp'] = address['PublicIp']
                res = ec2.associate_address(aws_retry=True, **params)
            except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
                msg = f"Couldn't associate Elastic IP address with instance '{device_id}'"
                module.fail_json_aws(e, msg=msg)
        else:
            params = dict(NetworkInterfaceId=device_id, AllocationId=address['AllocationId'], AllowReassociation=allow_reassociation)
            if private_ip_address:
                params['PrivateIpAddress'] = private_ip_address
            try:
                res = ec2.associate_address(aws_retry=True, **params)
            except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
                msg = f"Couldn't associate Elastic IP address with network interface '{device_id}'"
                module.fail_json_aws(e, msg=msg)
        if not res:
            module.fail_json(msg='Association failed.')
    return {'changed': True}