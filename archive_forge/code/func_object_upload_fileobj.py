from botocore.exceptions import ClientError
from boto3.s3.transfer import create_transfer_manager
from boto3.s3.transfer import TransferConfig, S3Transfer
from boto3.s3.transfer import ProgressCallbackInvoker
from boto3 import utils
def object_upload_fileobj(self, Fileobj, ExtraArgs=None, Callback=None, Config=None):
    """Upload a file-like object to this object.

    The file-like object must be in binary mode.

    This is a managed transfer which will perform a multipart upload in
    multiple threads if necessary.

    Usage::

        import boto3
        s3 = boto3.resource('s3')
        bucket = s3.Bucket('mybucket')
        obj = bucket.Object('mykey')

        with open('filename', 'rb') as data:
            obj.upload_fileobj(data)

    :type Fileobj: a file-like object
    :param Fileobj: A file-like object to upload. At a minimum, it must
        implement the `read` method, and must return bytes.

    :type ExtraArgs: dict
    :param ExtraArgs: Extra arguments that may be passed to the
        client operation.

    :type Callback: function
    :param Callback: A method which takes a number of bytes transferred to
        be periodically called during the upload.

    :type Config: boto3.s3.transfer.TransferConfig
    :param Config: The transfer configuration to be used when performing the
        upload.
    """
    return self.meta.client.upload_fileobj(Fileobj=Fileobj, Bucket=self.bucket_name, Key=self.key, ExtraArgs=ExtraArgs, Callback=Callback, Config=Config)