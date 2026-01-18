from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import os
import threading
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import upload_util
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def restart_download(self, start_byte):
    """Restarts the download_thread.

    Args:
      start_byte (int): The start byte for the new download call.
    """
    self.shutdown_event.set()
    with self.buffer_condition:
        self.buffer_condition.notify_all()
    self.wait_for_download_thread_to_terminate()
    self.buffer_queue.clear()
    self.shutdown_event.clear()
    self.start_download_thread(start_byte)