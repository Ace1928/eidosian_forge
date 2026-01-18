from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import csv
import datetime
import enum
import os
from googlecloudsdk.command_lib.storage import thread_messages
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def send_error_message(task_status_queue, source_resource, destination_resource, error):
    """Send ManifestMessage for failed copy to central processing."""
    _send_manifest_message(task_status_queue, source_resource, destination_resource, ResultStatus.ERROR, md5_hash=None, description=str(error))