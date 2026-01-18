from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.command_lib.transfer import jobs_flag_util
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
def generate_transfer_job_message(args, messages, existing_job=None):
    """Generates Apitools transfer message based on command arguments."""
    if existing_job:
        job = existing_job
    else:
        job = messages.TransferJob()
    if not job.projectId:
        job.projectId = properties.VALUES.core.project.Get()
    if getattr(args, 'name', None):
        job.name = name_util.add_job_prefix(args.name)
    if getattr(args, 'description', None):
        job.description = args.description
    if existing_job:
        if getattr(args, 'status', None):
            status_key = args.status.upper()
            job.status = getattr(messages.TransferJob.StatusValueValuesEnum, status_key)
    else:
        job.status = messages.TransferJob.StatusValueValuesEnum.ENABLED
    _create_or_modify_transfer_spec(job, args, messages)
    has_event_stream_flag = _create_or_modify_event_stream_configuration(job, args, messages)
    _create_or_modify_schedule(job, args, messages, is_update=bool(existing_job), has_event_stream_flag=has_event_stream_flag)
    _create_or_modify_notification_config(job, args, messages, is_update=bool(existing_job))
    _create_or_modify_logging_config(job, args, messages)
    if existing_job:
        return generate_patch_transfer_job_message(messages, job, UPDATE_FIELD_MASK)
    return job