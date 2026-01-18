from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.cognito.sync import exceptions
def describe_dataset(self, identity_pool_id, identity_id, dataset_name):
    """
        Gets metadata about a dataset by identity and dataset name.
        The credentials used to make this API call need to have access
        to the identity data. With Amazon Cognito Sync, each identity
        has access only to its own data. You should use Amazon Cognito
        Identity service to retrieve the credentials necessary to make
        this API call.

        :type identity_pool_id: string
        :param identity_pool_id: A name-spaced GUID (for example, us-
            east-1:23EC4050-6AEA-7089-A2DD-08002EXAMPLE) created by Amazon
            Cognito. GUID generation is unique within a region.

        :type identity_id: string
        :param identity_id: A name-spaced GUID (for example, us-
            east-1:23EC4050-6AEA-7089-A2DD-08002EXAMPLE) created by Amazon
            Cognito. GUID generation is unique within a region.

        :type dataset_name: string
        :param dataset_name: A string of up to 128 characters. Allowed
            characters are a-z, A-Z, 0-9, '_' (underscore), '-' (dash), and '.'
            (dot).

        """
    uri = '/identitypools/{0}/identities/{1}/datasets/{2}'.format(identity_pool_id, identity_id, dataset_name)
    return self.make_request('GET', uri, expected_status=200)