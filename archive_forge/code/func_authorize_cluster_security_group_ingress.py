import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def authorize_cluster_security_group_ingress(self, cluster_security_group_name, cidrip=None, ec2_security_group_name=None, ec2_security_group_owner_id=None):
    """
        Adds an inbound (ingress) rule to an Amazon Redshift security
        group. Depending on whether the application accessing your
        cluster is running on the Internet or an EC2 instance, you can
        authorize inbound access to either a Classless Interdomain
        Routing (CIDR) IP address range or an EC2 security group. You
        can add as many as 20 ingress rules to an Amazon Redshift
        security group.

        For an overview of CIDR blocks, see the Wikipedia article on
        `Classless Inter-Domain Routing`_.

        You must also associate the security group with a cluster so
        that clients running on these IP addresses or the EC2 instance
        are authorized to connect to the cluster. For information
        about managing security groups, go to `Working with Security
        Groups`_ in the Amazon Redshift Management Guide .

        :type cluster_security_group_name: string
        :param cluster_security_group_name: The name of the security group to
            which the ingress rule is added.

        :type cidrip: string
        :param cidrip: The IP range to be added the Amazon Redshift security
            group.

        :type ec2_security_group_name: string
        :param ec2_security_group_name: The EC2 security group to be added the
            Amazon Redshift security group.

        :type ec2_security_group_owner_id: string
        :param ec2_security_group_owner_id: The AWS account number of the owner
            of the security group specified by the EC2SecurityGroupName
            parameter. The AWS Access Key ID is not an acceptable value.
        Example: `111122223333`

        """
    params = {'ClusterSecurityGroupName': cluster_security_group_name}
    if cidrip is not None:
        params['CIDRIP'] = cidrip
    if ec2_security_group_name is not None:
        params['EC2SecurityGroupName'] = ec2_security_group_name
    if ec2_security_group_owner_id is not None:
        params['EC2SecurityGroupOwnerId'] = ec2_security_group_owner_id
    return self._make_request(action='AuthorizeClusterSecurityGroupIngress', verb='POST', path='/', params=params)