import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import add_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
@staticmethod
def do_check_mode(module, connection, _image_id):
    image = connection.describe_images(Filters=[{'Name': 'name', 'Values': [str(module.params['name'])]}])
    if not image['Images']:
        module.exit_json(changed=True, msg='Would have created a AMI if not in check mode.')
    else:
        module.exit_json(changed=False, msg='Error registering image: AMI name is already in use by another AMI')