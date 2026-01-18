from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
def add_source_creds_flag(parser):
    parser.add_argument('--source-creds-file', help='Path to a local file on your machine that includes credentials for an Amazon S3 or Azure Blob Storage source (not required for Google Cloud Storage sources). If not specified for an S3 source, gcloud will check your system for an AWS config file. However, this flag must be specified to use AWS\'s "role_arn" auth service. For formatting, see:\n\nS3: https://cloud.google.com/storage-transfer/docs/reference/rest/v1/TransferSpec#AwsAccessKey\nNote: Be sure to put quotations around the JSON value strings.\n\nAzure: https://cloud.google.com/storage-transfer/docs/reference/rest/v1/TransferSpec#AzureCredentials\n\n')