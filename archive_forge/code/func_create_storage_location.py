import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def create_storage_location(self):
    """
        Creates the Amazon S3 storage location for the account.  This
        location is used to store user log files.

        :raises: TooManyBucketsException,
                 S3SubscriptionRequiredException,
                 InsufficientPrivilegesException

        """
    return self._get_response('CreateStorageLocation', params={})