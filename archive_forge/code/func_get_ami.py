from __future__ import absolute_import, division, print_function
import traceback
import uuid
import time
import base64
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def get_ami(self):
    """
        Get AWS EC2 Image
        :return:
            Latest AMI
        """
    instance_ami = None
    client = boto3.client('ec2', region_name=self.parameters['region'])
    try:
        instance_ami = client.describe_images(Filters=[{'Name': 'name', 'Values': [self.rest_api.environment_data['AMI_FILTER']]}], Owners=[self.rest_api.environment_data['AWS_ACCOUNT']])
    except ClientError as error:
        self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
    latest_date = instance_ami['Images'][0]['CreationDate']
    latest_ami = instance_ami['Images'][0]['ImageId']
    for image in instance_ami['Images']:
        if image['CreationDate'] > latest_date:
            latest_date = image['CreationDate']
            latest_ami = image['ImageId']
    return latest_ami