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
def get_temporary_gzipped_file(file_path):
    zipped_file_path = file_path + storage_url.TEMPORARY_FILE_SUFFIX
    with files.BinaryFileReader(file_path) as file_reader:
        with gzip.open(zipped_file_path, 'wb') as gzip_file_writer:
            shutil.copyfileobj(file_reader, gzip_file_writer)
    return zipped_file_path