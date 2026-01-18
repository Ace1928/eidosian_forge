import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def create_hsm(self, subnet_id, ssh_key, iam_role_arn, subscription_type, eni_ip=None, external_id=None, client_token=None, syslog_ip=None):
    """
        Creates an uninitialized HSM instance. Running this command
        provisions an HSM appliance and will result in charges to your
        AWS account for the HSM.

        :type subnet_id: string
        :param subnet_id: The identifier of the subnet in your VPC in which to
            place the HSM.

        :type ssh_key: string
        :param ssh_key: The SSH public key to install on the HSM.

        :type eni_ip: string
        :param eni_ip: The IP address to assign to the HSM's ENI.

        :type iam_role_arn: string
        :param iam_role_arn: The ARN of an IAM role to enable the AWS CloudHSM
            service to allocate an ENI on your behalf.

        :type external_id: string
        :param external_id: The external ID from **IamRoleArn**, if present.

        :type subscription_type: string
        :param subscription_type: The subscription type.

        :type client_token: string
        :param client_token: A user-defined token to ensure idempotence.
            Subsequent calls to this action with the same token will be
            ignored.

        :type syslog_ip: string
        :param syslog_ip: The IP address for the syslog monitoring server.

        """
    params = {'SubnetId': subnet_id, 'SshKey': ssh_key, 'IamRoleArn': iam_role_arn, 'SubscriptionType': subscription_type}
    if eni_ip is not None:
        params['EniIp'] = eni_ip
    if external_id is not None:
        params['ExternalId'] = external_id
    if client_token is not None:
        params['ClientToken'] = client_token
    if syslog_ip is not None:
        params['SyslogIp'] = syslog_ip
    return self.make_request(action='CreateHsm', body=json.dumps(params))