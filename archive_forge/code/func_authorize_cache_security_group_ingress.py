import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def authorize_cache_security_group_ingress(self, cache_security_group_name, ec2_security_group_name, ec2_security_group_owner_id):
    """
        The AuthorizeCacheSecurityGroupIngress operation allows
        network ingress to a cache security group. Applications using
        ElastiCache must be running on Amazon EC2, and Amazon EC2
        security groups are used as the authorization mechanism.
        You cannot authorize ingress from an Amazon EC2 security group
        in one Region to an ElastiCache cluster in another Region.

        :type cache_security_group_name: string
        :param cache_security_group_name: The cache security group which will
            allow network ingress.

        :type ec2_security_group_name: string
        :param ec2_security_group_name: The Amazon EC2 security group to be
            authorized for ingress to the cache security group.

        :type ec2_security_group_owner_id: string
        :param ec2_security_group_owner_id: The AWS account number of the
            Amazon EC2 security group owner. Note that this is not the same
            thing as an AWS access key ID - you must provide a valid AWS
            account number for this parameter.

        """
    params = {'CacheSecurityGroupName': cache_security_group_name, 'EC2SecurityGroupName': ec2_security_group_name, 'EC2SecurityGroupOwnerId': ec2_security_group_owner_id}
    return self._make_request(action='AuthorizeCacheSecurityGroupIngress', verb='POST', path='/', params=params)