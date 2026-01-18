from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import k8s_object
@force_override.setter
def force_override(self, value):
    self.spec.forceOverride = value