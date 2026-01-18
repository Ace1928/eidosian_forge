from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
def block_until_operation_created(name):
    """Blocks until job creates an operation and returns operation name."""
    with progress_tracker.ProgressTracker(message='Polling for latest operation name'):
        return retry.Retryer().RetryOnResult(api_get, args=[name], should_retry_if=_has_not_created_operation, sleep_ms=properties.VALUES.transfer.no_async_polling_interval_ms.GetInt()).latestOperationName