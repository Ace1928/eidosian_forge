from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from six.moves import queue
def object_iterator(self):
    return self._resource_iterator(self._object_delete_tasks)