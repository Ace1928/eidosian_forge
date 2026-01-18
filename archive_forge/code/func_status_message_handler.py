from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import datetime
import enum
import threading
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import metrics_util
from googlecloudsdk.command_lib.storage import thread_messages
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import scaled_integer
import six
def status_message_handler(task_status_queue, status_tracker):
    """Thread method for submiting items from queue to tracker for processing."""
    unhandled_message_exists = False
    while True:
        status_message = task_status_queue.get()
        if status_message == '_SHUTDOWN':
            break
        if status_tracker:
            status_tracker.add_message(status_message)
        else:
            unhandled_message_exists = True
    if unhandled_message_exists:
        log.warning('Status message submitted to task_status_queue without a manager to print it.')