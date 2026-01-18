from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def create_symlink_from_temporary_placeholder(placeholder_path, symlink_path):
    symlink_target = files.ReadFileContents(placeholder_path)
    os.symlink(symlink_target, symlink_path)