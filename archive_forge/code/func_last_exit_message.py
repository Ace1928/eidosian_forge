from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.core.console import console_attr
@property
def last_exit_message(self):
    if self.status.lastAttemptResult is not None and self.status.lastAttemptResult.status.message is not None:
        return self.status.lastAttemptResult.status.message
    return ''