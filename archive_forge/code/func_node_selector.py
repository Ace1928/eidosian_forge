from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
@property
def node_selector(self):
    """The node selector as a dictionary { accelerator_type: value}."""
    self._EnsureNodeSelector()
    return k8s_object.KeyValueListAsDictionaryWrapper(self.spec.nodeSelector.additionalProperties, self._messages.RevisionSpec.NodeSelectorValue.AdditionalProperty, key_field='key', value_field='value')