from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.alloydb import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def normalize_automated_backup_policy(policy):
    """Normalizes the policy so that it looks correct when printed."""
    if policy is None:
        return
    if policy.weeklySchedule is None:
        return
    for start_time in policy.weeklySchedule.startTimes:
        if start_time.hours is None:
            start_time.hours = 0