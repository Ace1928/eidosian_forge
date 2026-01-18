from botocore.exceptions import ClientError
from boto3.s3.transfer import create_transfer_manager
from boto3.s3.transfer import TransferConfig, S3Transfer
from boto3.s3.transfer import ProgressCallbackInvoker
from boto3 import utils
def inject_object_methods(class_attributes, **kwargs):
    utils.inject_attribute(class_attributes, 'upload_file', object_upload_file)
    utils.inject_attribute(class_attributes, 'download_file', object_download_file)
    utils.inject_attribute(class_attributes, 'copy', object_copy)
    utils.inject_attribute(class_attributes, 'upload_fileobj', object_upload_fileobj)
    utils.inject_attribute(class_attributes, 'download_fileobj', object_download_fileobj)