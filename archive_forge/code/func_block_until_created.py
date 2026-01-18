from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
def block_until_created(name):
    """Blocks until agent pool is created. Useful for scripting."""
    with progress_tracker.ProgressTracker(message='Waiting for backend to create agent pool'):
        result = retry.Retryer().RetryOnResult(api_get, args=[name], should_retry_if=_is_agent_pool_still_creating, sleep_ms=properties.VALUES.transfer.no_async_polling_interval_ms.GetInt())
    return result