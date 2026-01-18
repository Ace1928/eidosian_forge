from botocore.exceptions import ClientError
from boto3.s3.transfer import create_transfer_manager
from boto3.s3.transfer import TransferConfig, S3Transfer
from boto3.s3.transfer import ProgressCallbackInvoker
from boto3 import utils
def object_summary_load(self, *args, **kwargs):
    """
    Calls s3.Client.head_object to update the attributes of the ObjectSummary
    resource.
    """
    response = self.meta.client.head_object(Bucket=self.bucket_name, Key=self.key)
    if 'ContentLength' in response:
        response['Size'] = response.pop('ContentLength')
    self.meta.data = response