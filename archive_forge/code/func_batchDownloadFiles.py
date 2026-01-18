from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import tempfile
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import download_util
from googlecloudsdk.command_lib.artifacts import file_util
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.core import log
def batchDownloadFiles(self, args, repo_ref, list_files):
    for files in list_files:
        file_id = os.path.basename(files.name)
        file_name = file_id.rsplit(':', 1)[1].replace('%2F', '/')
        if '/' in file_name:
            d = os.path.dirname(file_name)
            os.makedirs(os.path.join(args.destination, d), exist_ok=True)
        self.downloadGenericArtifact(args, repo_ref, file_id, file_name)