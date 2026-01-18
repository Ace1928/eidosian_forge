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
def detach_eni(connection, eni, module):
    if module.check_mode:
        module.exit_json(changed=True, msg='Would have detached ENI if not in check mode.')
    eni_id = eni['NetworkInterfaceId']
    force_detach = module.params.get('force_detach')
    if 'Attachment' in eni:
        connection.detach_network_interface(aws_retry=True, AttachmentId=eni['Attachment']['AttachmentId'], Force=force_detach)
        _wait_for_detach(connection, module, eni_id)
        return True
    return False