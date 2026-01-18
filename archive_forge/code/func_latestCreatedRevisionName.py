from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.kuberun import structuredout
from googlecloudsdk.command_lib.kuberun import kubernetes_consts as k8s
@property
def latestCreatedRevisionName(self):
    return self._props.get(k8s.FIELD_LATEST_CREATED_REVISION_NAME)