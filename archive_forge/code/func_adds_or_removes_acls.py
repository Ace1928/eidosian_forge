from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.core.util import debug_output
def adds_or_removes_acls(user_request_args):
    """Returns whether existing ACL policy needs to be patched."""
    return bool(user_request_args.resource_args and (user_request_args.resource_args.acl_grants_to_add or user_request_args.resource_args.acl_grants_to_remove or getattr(user_request_args.resource_args, 'default_object_acl_grants_to_add', False) or getattr(user_request_args.resource_args, 'default_object_acl_grants_to_remove', False)))