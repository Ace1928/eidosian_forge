from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.core.util import retry
def retryIf(exc_type, exc_value, unused_traceback, unused_state):
    return exc_type == exceptions.HttpError and exc_value.status_code == status