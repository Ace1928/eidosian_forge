from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.command_lib.storage import thread_messages
def increment_count_callback(status_queue):
    status_queue.put(thread_messages.IncrementProgressMessage())