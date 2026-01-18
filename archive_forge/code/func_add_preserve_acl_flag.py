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
def add_preserve_acl_flag(parser, hidden=False):
    """Adds preserve ACL flag."""
    parser.add_argument('--preserve-acl', '-p', action=arg_parsers.StoreTrueFalseAction, hidden=hidden, help='Preserves ACLs when copying in the cloud. This option is Cloud Storage-only, and you need OWNER access to all copied objects. If all objects in the destination bucket should have the same ACL, you can also set a default object ACL on that bucket instead of using this flag.\nPreserving ACLs is the default behavior for updating existing objects.')