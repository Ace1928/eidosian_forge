from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def get_preserve_symlink_from_user_request(user_request_args):
    return user_request_args.preserve_symlinks if user_request_args else None