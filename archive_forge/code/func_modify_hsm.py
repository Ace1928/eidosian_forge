import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def modify_hsm(self, hsm_arn, subnet_id=None, eni_ip=None, iam_role_arn=None, external_id=None, syslog_ip=None):
    """
        Modifies an HSM.

        :type hsm_arn: string
        :param hsm_arn: The ARN of the HSM to modify.

        :type subnet_id: string
        :param subnet_id: The new identifier of the subnet that the HSM is in.

        :type eni_ip: string
        :param eni_ip: The new IP address for the elastic network interface
            attached to the HSM.

        :type iam_role_arn: string
        :param iam_role_arn: The new IAM role ARN.

        :type external_id: string
        :param external_id: The new external ID.

        :type syslog_ip: string
        :param syslog_ip: The new IP address for the syslog monitoring server.

        """
    params = {'HsmArn': hsm_arn}
    if subnet_id is not None:
        params['SubnetId'] = subnet_id
    if eni_ip is not None:
        params['EniIp'] = eni_ip
    if iam_role_arn is not None:
        params['IamRoleArn'] = iam_role_arn
    if external_id is not None:
        params['ExternalId'] = external_id
    if syslog_ip is not None:
        params['SyslogIp'] = syslog_ip
    return self.make_request(action='ModifyHsm', body=json.dumps(params))