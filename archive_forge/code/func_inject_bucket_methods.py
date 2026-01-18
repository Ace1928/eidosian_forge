from botocore.exceptions import ClientError
from boto3.s3.transfer import create_transfer_manager
from boto3.s3.transfer import TransferConfig, S3Transfer
from boto3.s3.transfer import ProgressCallbackInvoker
from boto3 import utils
def inject_bucket_methods(class_attributes, **kwargs):
    utils.inject_attribute(class_attributes, 'load', bucket_load)
    utils.inject_attribute(class_attributes, 'upload_file', bucket_upload_file)
    utils.inject_attribute(class_attributes, 'download_file', bucket_download_file)
    utils.inject_attribute(class_attributes, 'copy', bucket_copy)
    utils.inject_attribute(class_attributes, 'upload_fileobj', bucket_upload_fileobj)
    utils.inject_attribute(class_attributes, 'download_fileobj', bucket_download_fileobj)