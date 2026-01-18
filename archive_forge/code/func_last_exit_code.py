from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.core.console import console_attr
@property
def last_exit_code(self):
    if self.status.lastAttemptResult is not None and self.status.lastAttemptResult.exitCode is not None:
        return self.status.lastAttemptResult.exitCode
    elif self.status.completionTime is not None:
        return 0
    return None