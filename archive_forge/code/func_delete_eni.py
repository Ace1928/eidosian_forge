import time
from ipaddress import ip_address
from ipaddress import ip_network
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def delete_eni(connection, module):
    eni = uniquely_find_eni(connection, module)
    if not eni:
        module.exit_json(changed=False)
    if module.check_mode:
        module.exit_json(changed=True, msg='Would have deleted ENI if not in check mode.')
    eni_id = eni['NetworkInterfaceId']
    force_detach = module.params.get('force_detach')
    try:
        if force_detach is True:
            if 'Attachment' in eni:
                connection.detach_network_interface(aws_retry=True, AttachmentId=eni['Attachment']['AttachmentId'], Force=True)
                _wait_for_detach(connection, module, eni_id)
            connection.delete_network_interface(aws_retry=True, NetworkInterfaceId=eni_id)
            changed = True
        else:
            connection.delete_network_interface(aws_retry=True, NetworkInterfaceId=eni_id)
            changed = True
        module.exit_json(changed=changed)
    except is_boto3_error_code('InvalidNetworkInterfaceID.NotFound'):
        module.exit_json(changed=False)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, f'Failure during delete of {eni_id}')