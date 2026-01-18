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
def add_soft_deleted_flag(parser):
    """Adds flag for only displaying soft-deleted objects."""
    parser.add_argument('--soft-deleted', action='store_true', help='Displays soft-deleted objects only. Excludes live and noncurrent objects.')