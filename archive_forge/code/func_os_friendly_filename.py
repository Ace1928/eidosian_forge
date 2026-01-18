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
def os_friendly_filename(self, file_id):
    filename = file_id.replace(':', '%3A')
    filename = filename.replace('\\', '%5C')
    filename = filename.replace('*', '%3F')
    filename = filename.replace('?', '%22')
    filename = filename.replace('<', '%3C')
    filename = filename.replace('>', '%2E')
    filename = filename.replace('|', '%7C')
    return filename