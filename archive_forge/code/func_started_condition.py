from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
@property
def started_condition(self):
    if self.conditions and STARTED_CONDITION in self.conditions:
        return self.conditions[STARTED_CONDITION]