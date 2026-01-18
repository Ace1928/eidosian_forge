import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cognito.identity import exceptions
def delete_identity_pool(self, identity_pool_id):
    """
        Deletes a user pool. Once a pool is deleted, users will not be
        able to authenticate with the pool.

        :type identity_pool_id: string
        :param identity_pool_id: An identity pool ID in the format REGION:GUID.

        """
    params = {'IdentityPoolId': identity_pool_id}
    return self.make_request(action='DeleteIdentityPool', body=json.dumps(params))