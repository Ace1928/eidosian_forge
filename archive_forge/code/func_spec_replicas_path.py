from pprint import pformat
from six import iteritems
import re
@spec_replicas_path.setter
def spec_replicas_path(self, spec_replicas_path):
    """
        Sets the spec_replicas_path of this
        V1beta1CustomResourceSubresourceScale.
        SpecReplicasPath defines the JSON path inside of a CustomResource that
        corresponds to Scale.Spec.Replicas. Only JSON paths without the array
        notation are allowed. Must be a JSON Path under .spec. If there is no
        value under the given path in the CustomResource, the /scale subresource
        will return an error on GET.

        :param spec_replicas_path: The spec_replicas_path of this
        V1beta1CustomResourceSubresourceScale.
        :type: str
        """
    if spec_replicas_path is None:
        raise ValueError('Invalid value for `spec_replicas_path`, must not be `None`')
    self._spec_replicas_path = spec_replicas_path