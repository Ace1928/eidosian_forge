from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.transfer import jobs_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import scaled_integer
def display_monitoring_view(name):
    """Prints and updates operation statistics, blocking until copy complete."""
    initial_operation = api_get(name)
    initial_metadata = encoding.MessageToDict(initial_operation.metadata)
    operation_name = name_util.remove_operation_prefix(initial_operation.name)
    log.status.Print('Operation name: ' + operation_name)
    log.status.Print('Parent job: ' + name_util.remove_job_prefix(initial_metadata['transferJobName']))
    if 'startTime' in initial_metadata:
        log.status.Print('Start time: ' + initial_metadata['startTime'])
    final_operation = _poll_progress(initial_operation.name)
    final_metadata = encoding.MessageToDict(final_operation.metadata)
    if 'endTime' in final_metadata:
        log.status.Print('\nEnd time: ' + final_metadata['endTime'])
    if 'errorBreakdowns' in final_metadata:
        describe_command = 'gcloud transfer operations describe ' + operation_name
        log.status.Print('\nTo investigate errors, run: \n{}\n'.format(describe_command))