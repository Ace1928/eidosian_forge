from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
@property
def gcs_location(self):
    return self._m.status.gcs.location