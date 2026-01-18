import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def revoke_db_security_group_ingress(self, db_security_group_name, cidrip=None, ec2_security_group_name=None, ec2_security_group_id=None, ec2_security_group_owner_id=None):
    """
        Revokes ingress from a DBSecurityGroup for previously
        authorized IP ranges or EC2 or VPC Security Groups. Required
        parameters for this API are one of CIDRIP, EC2SecurityGroupId
        for VPC, or (EC2SecurityGroupOwnerId and either
        EC2SecurityGroupName or EC2SecurityGroupId).

        :type db_security_group_name: string
        :param db_security_group_name: The name of the DB security group to
            revoke ingress from.

        :type cidrip: string
        :param cidrip: The IP range to revoke access from. Must be a valid CIDR
            range. If `CIDRIP` is specified, `EC2SecurityGroupName`,
            `EC2SecurityGroupId` and `EC2SecurityGroupOwnerId` cannot be
            provided.

        :type ec2_security_group_name: string
        :param ec2_security_group_name: The name of the EC2 security group to
            revoke access from. For VPC DB security groups,
            `EC2SecurityGroupId` must be provided. Otherwise,
            EC2SecurityGroupOwnerId and either `EC2SecurityGroupName` or
            `EC2SecurityGroupId` must be provided.

        :type ec2_security_group_id: string
        :param ec2_security_group_id: The id of the EC2 security group to
            revoke access from. For VPC DB security groups,
            `EC2SecurityGroupId` must be provided. Otherwise,
            EC2SecurityGroupOwnerId and either `EC2SecurityGroupName` or
            `EC2SecurityGroupId` must be provided.

        :type ec2_security_group_owner_id: string
        :param ec2_security_group_owner_id: The AWS Account Number of the owner
            of the EC2 security group specified in the `EC2SecurityGroupName`
            parameter. The AWS Access Key ID is not an acceptable value. For
            VPC DB security groups, `EC2SecurityGroupId` must be provided.
            Otherwise, EC2SecurityGroupOwnerId and either
            `EC2SecurityGroupName` or `EC2SecurityGroupId` must be provided.

        """
    params = {'DBSecurityGroupName': db_security_group_name}
    if cidrip is not None:
        params['CIDRIP'] = cidrip
    if ec2_security_group_name is not None:
        params['EC2SecurityGroupName'] = ec2_security_group_name
    if ec2_security_group_id is not None:
        params['EC2SecurityGroupId'] = ec2_security_group_id
    if ec2_security_group_owner_id is not None:
        params['EC2SecurityGroupOwnerId'] = ec2_security_group_owner_id
    return self._make_request(action='RevokeDBSecurityGroupIngress', verb='POST', path='/', params=params)