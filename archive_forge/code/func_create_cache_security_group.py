import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def create_cache_security_group(self, cache_security_group_name, description):
    """
        The CreateCacheSecurityGroup operation creates a new cache
        security group. Use a cache security group to control access
        to one or more cache clusters.

        Cache security groups are only used when you are creating a
        cluster outside of an Amazon Virtual Private Cloud (VPC). If
        you are creating a cluster inside of a VPC, use a cache subnet
        group instead. For more information, see
        CreateCacheSubnetGroup .

        :type cache_security_group_name: string
        :param cache_security_group_name: A name for the cache security group.
            This value is stored as a lowercase string.
        Constraints: Must contain no more than 255 alphanumeric characters.
            Must not be the word "Default".

        Example: `mysecuritygroup`

        :type description: string
        :param description: A description for the cache security group.

        """
    params = {'CacheSecurityGroupName': cache_security_group_name, 'Description': description}
    return self._make_request(action='CreateCacheSecurityGroup', verb='POST', path='/', params=params)