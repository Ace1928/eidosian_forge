import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def authorize_db_security_group_ingress(self, db_security_group_name, cidrip=None, ec2_security_group_name=None, ec2_security_group_id=None, ec2_security_group_owner_id=None):
    """
        Enables ingress to a DBSecurityGroup using one of two forms of
        authorization. First, EC2 or VPC security groups can be added
        to the DBSecurityGroup if the application using the database
        is running on EC2 or VPC instances. Second, IP ranges are
        available if the application accessing your database is
        running on the Internet. Required parameters for this API are
        one of CIDR range, EC2SecurityGroupId for VPC, or
        (EC2SecurityGroupOwnerId and either EC2SecurityGroupName or
        EC2SecurityGroupId for non-VPC).
        You cannot authorize ingress from an EC2 security group in one
        Region to an Amazon RDS DB instance in another. You cannot
        authorize ingress from a VPC security group in one VPC to an
        Amazon RDS DB instance in another.
        For an overview of CIDR ranges, go to the `Wikipedia
        Tutorial`_.

        :type db_security_group_name: string
        :param db_security_group_name: The name of the DB security group to add
            authorization to.

        :type cidrip: string
        :param cidrip: The IP range to authorize.

        :type ec2_security_group_name: string
        :param ec2_security_group_name: Name of the EC2 security group to
            authorize. For VPC DB security groups, `EC2SecurityGroupId` must be
            provided. Otherwise, EC2SecurityGroupOwnerId and either
            `EC2SecurityGroupName` or `EC2SecurityGroupId` must be provided.

        :type ec2_security_group_id: string
        :param ec2_security_group_id: Id of the EC2 security group to
            authorize. For VPC DB security groups, `EC2SecurityGroupId` must be
            provided. Otherwise, EC2SecurityGroupOwnerId and either
            `EC2SecurityGroupName` or `EC2SecurityGroupId` must be provided.

        :type ec2_security_group_owner_id: string
        :param ec2_security_group_owner_id: AWS Account Number of the owner of
            the EC2 security group specified in the EC2SecurityGroupName
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
    return self._make_request(action='AuthorizeDBSecurityGroupIngress', verb='POST', path='/', params=params)