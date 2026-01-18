from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task_util
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def fix_user_agent_for_gsutil_shim():
    """Transform the user agent when the gsutil shim is used to run gcloud.

  This transforms `gcloud.storage.command` to `gcloud.gsutil.command`.

  This needs to be called by every command, so the best place to put this is
  likely surface/storage/__init__.py, but if there's a better place it could be
  called there instead.
  """
    if properties.VALUES.storage.run_by_gsutil_shim.GetBool():
        command_path_string = properties.VALUES.metrics.command_name.Get().replace('gcloud.storage.', 'gcloud.gslibshim.')
        properties.VALUES.SetInvocationValue(properties.VALUES.metrics.command_name, command_path_string, None)