import boto
from boto.gs.bucket import Bucket
from boto.s3.connection import S3Connection
from boto.s3.connection import SubdomainCallingFormat
from boto.s3.connection import check_lowercase_bucketname
from boto.compat import six
from boto.utils import get_utf8able_str
def get_bucket(self, bucket_name, validate=True, headers=None):
    """
        Retrieves a bucket by name.

        If the bucket does not exist, an ``S3ResponseError`` will be raised. If
        you are unsure if the bucket exists or not, you can use the
        ``S3Connection.lookup`` method, which will either return a valid bucket
        or ``None``.

        :type bucket_name: string
        :param bucket_name: The name of the bucket

        :type headers: dict
        :param headers: Additional headers to pass along with the request to
            AWS.

        :type validate: boolean
        :param validate: If ``True``, it will try to fetch all keys within the
            given bucket. (Default: ``True``)
        """
    bucket = self.bucket_class(self, bucket_name)
    if validate:
        bucket.get_all_keys(headers, maxkeys=0)
    return bucket