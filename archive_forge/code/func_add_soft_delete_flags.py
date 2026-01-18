from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def add_soft_delete_flags(parser):
    """Adds flags related to soft delete feature."""
    add_soft_deleted_flag(parser)
    parser.add_argument('--exhaustive', action='store_true', help='For features like soft delete, the API may return an empty list. If present, continue querying. This may incur costs from repeated LIST calls and may not return any additional objects.')
    parser.add_argument('--next-page-token', help='Page token for resuming LIST calls.')