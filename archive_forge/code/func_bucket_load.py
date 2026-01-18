from botocore.exceptions import ClientError
from boto3.s3.transfer import create_transfer_manager
from boto3.s3.transfer import TransferConfig, S3Transfer
from boto3.s3.transfer import ProgressCallbackInvoker
from boto3 import utils
def bucket_load(self, *args, **kwargs):
    """
    Calls s3.Client.list_buckets() to update the attributes of the Bucket
    resource.
    """
    self.meta.data = {}
    try:
        response = self.meta.client.list_buckets()
        for bucket_data in response['Buckets']:
            if bucket_data['Name'] == self.name:
                self.meta.data = bucket_data
                break
    except ClientError as e:
        if not e.response.get('Error', {}).get('Code') == 'AccessDenied':
            raise