from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import gzip
import os
import shutil
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def should_gzip_locally(gzip_settings, file_path):
    return _should_gzip_file_type(gzip_settings, file_path) == user_request_args_factory.GzipType.LOCAL