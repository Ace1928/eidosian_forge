from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import properties
def remove_job_prefix(job_string):
    """Removes prefix from transfer job if necessary."""
    if job_string.startswith(_JOBS_PREFIX_STRING):
        return job_string[len(_JOBS_PREFIX_STRING):]
    return job_string