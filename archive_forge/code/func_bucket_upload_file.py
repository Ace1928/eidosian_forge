from botocore.exceptions import ClientError
from boto3.s3.transfer import create_transfer_manager
from boto3.s3.transfer import TransferConfig, S3Transfer
from boto3.s3.transfer import ProgressCallbackInvoker
from boto3 import utils
def bucket_upload_file(self, Filename, Key, ExtraArgs=None, Callback=None, Config=None):
    """Upload a file to an S3 object.

    Usage::

        import boto3
        s3 = boto3.resource('s3')
        s3.Bucket('mybucket').upload_file('/tmp/hello.txt', 'hello.txt')

    Similar behavior as S3Transfer's upload_file() method,
    except that parameters are capitalized. Detailed examples can be found at
    :ref:`S3Transfer's Usage <ref_s3transfer_usage>`.

    :type Filename: str
    :param Filename: The path to the file to upload.

    :type Key: str
    :param Key: The name of the key to upload to.

    :type ExtraArgs: dict
    :param ExtraArgs: Extra arguments that may be passed to the
        client operation.

    :type Callback: function
    :param Callback: A method which takes a number of bytes transferred to
        be periodically called during the upload.

    :type Config: boto3.s3.transfer.TransferConfig
    :param Config: The transfer configuration to be used when performing the
        transfer.
    """
    return self.meta.client.upload_file(Filename=Filename, Bucket=self.name, Key=Key, ExtraArgs=ExtraArgs, Callback=Callback, Config=Config)